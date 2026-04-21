"""Validation gates for fused_mlp_up.

Three gates must all pass before committing:
  1. Numerical correctness (within bf16 cast noise)
  2. Gradient correctness (within bf16 matmul noise)
  3. torch.compile(fullgraph=True, dynamic=False) compatibility

Run from repo root:
    python experiments/test_fused_mlp_up.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Triton on Windows needs an explicit C compiler pointer (same as harness.py)
os.environ.setdefault(
    "CC",
    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe",
)

import torch
import torch.nn.functional as F

from fused_kernels import fused_mlp_up


def _reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference: relu(F.linear(x, w)).square().

    Matches CastedLinear's cast-at-matmul semantics by casting w to x.dtype
    before the linear call. This mirrors what CastedLinear.forward does in
    the model.
    """
    return torch.relu(F.linear(x, w.to(x.dtype))).square()


def gate_numerical() -> None:
    torch.manual_seed(1337)
    # pr-1493 per-rank target: M = B*T = 8192, K = 512, N = 1024
    B, T, K, N = 8, 1024, 512, 1024
    device = torch.device("cuda")

    # Weight scale matches nn.Linear default kaiming_uniform (fan_in=K=512):
    # bound = sqrt(6 / (1+a^2) / fan_in) ~ 0.108; std ~ 0.062.
    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06

    ref = _reference(x, w)
    fused = fused_mlp_up(x, w)

    assert ref.shape == fused.shape, f"shape mismatch: ref {ref.shape}, fused {fused.shape}"
    assert fused.dtype == torch.bfloat16

    # bf16 has ~7-bit mantissa -> relative precision ~7e-3 per op. For a 512-term
    # reduction with differing tile/reduction order between cuBLAS and Triton, a
    # relative tolerance of ~1e-2 is the right gate. cos similarity > 0.9999 is
    # the secondary check capturing "direction-preserving" agreement.
    ref_f = ref.float()
    fused_f = fused.float()
    diff = (ref_f - fused_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_ref = ref_f.abs().max().item()
    rel_max = max_diff / max(max_ref, 1e-30)
    cos = F.cosine_similarity(ref_f.flatten(), fused_f.flatten(), dim=0).item()

    # Primary gate: cos similarity. bf16 storage granularity alone can produce
    # max_diff up to ~0.5 on values near 64 (mantissa gap), so absolute-tolerance
    # checks are the wrong tool for a K=512 bf16 GEMM comparison across two
    # different tile strategies. Cos > 0.9999 is the ForgeFuse Phase 2A bar.
    if cos < 0.9999:
        raise AssertionError(f"Gate 1 FAIL: forward cos {cos:.6f} < 0.9999")
    # Secondary: relative-to-max error should be under ~1% (bf16 relative precision
    # compounded over K=512 reduction is ~sqrt(512) * 2^-8 ~ 9e-2 worst-case; in
    # practice well below 1e-2).
    if rel_max > 2e-2:
        raise AssertionError(
            f"Gate 1 FAIL: forward rel_max {rel_max:.3e} > 2e-2 "
            f"(max_diff {max_diff:.3e}, max_ref {max_ref:.3e})"
        )
    print(
        f"[PASS] Gate 1 numerical: max_diff {max_diff:.2e} over max_ref {max_ref:.2e} "
        f"(rel {rel_max:.2e}), mean_diff {mean_diff:.2e}, cos {cos:.6f}"
    )


def gate_gradient() -> None:
    torch.manual_seed(1337)
    B, T, K, N = 4, 512, 512, 1024  # smaller shape for faster grad test
    device = torch.device("cuda")

    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06
    w.requires_grad_(True)

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)

    ref_out = _reference(x, w)
    (ref_out.float().pow(2).mean()).backward()

    fused_out = fused_mlp_up(x2, w2)
    (fused_out.float().pow(2).mean()).backward()

    assert x.grad is not None and x2.grad is not None, "grad_x not populated"
    assert w.grad is not None and w2.grad is not None, "grad_w not populated"

    gx_diff = (x.grad.float() - x2.grad.float()).abs()
    gw_diff = (w.grad - w2.grad).abs()
    gx_max = gx_diff.max().item()
    gw_max = gw_diff.max().item()
    gx_rel = gx_diff.sum().item() / x.grad.float().abs().sum().clamp_min(1e-30).item()
    gw_rel = gw_diff.sum().item() / w.grad.abs().sum().clamp_min(1e-30).item()

    if not torch.allclose(x.grad.float(), x2.grad.float(), atol=1e-4, rtol=1e-3):
        raise AssertionError(
            f"Gate 2 FAIL: grad_x max {gx_max:.2e}, rel L1 {gx_rel:.2e}"
        )
    if not torch.allclose(w.grad, w2.grad, atol=1e-4, rtol=1e-3):
        raise AssertionError(
            f"Gate 2 FAIL: grad_w max {gw_max:.2e}, rel L1 {gw_rel:.2e}"
        )
    print(
        f"[PASS] Gate 2 gradients: grad_x max {gx_max:.2e} (rel {gx_rel:.2e}), "
        f"grad_w max {gw_max:.2e} (rel {gw_rel:.2e})"
    )


def gate_compile() -> None:
    torch.manual_seed(1337)
    B, T, K, N = 4, 1024, 512, 1024
    device = torch.device("cuda")

    class MlpUpMod(torch.nn.Module):
        def __init__(self, K: int, N: int) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(N, K, dtype=torch.float32) * 0.06)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return fused_mlp_up(x, self.w)

    m = MlpUpMod(K, N).to(device)

    try:
        compiled = torch.compile(m, fullgraph=True, dynamic=False)
    except Exception as e:  # pragma: no cover — diagnostic only
        raise AssertionError(f"Gate 3 FAIL (compile call): {e}") from e

    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    try:
        out = compiled(x)
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Gate 3 FAIL (compiled forward): {e}") from e

    assert out.shape == (B, T, N), f"shape {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in compiled forward output"

    try:
        out.sum().backward()
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Gate 3 FAIL (compiled backward): {e}") from e

    assert m.w.grad is not None and torch.isfinite(m.w.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[PASS] Gate 3 compile: fullgraph=True fwd+bwd succeeded")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gate_numerical()
    gate_gradient()
    gate_compile()
    print("\n[ALL GATES PASS] fused_mlp_up is correct, diff-safe, compile-safe")


if __name__ == "__main__":
    main()
