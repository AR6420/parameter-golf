"""Validation gates for Phase 4B Triton W8A8 kernel.

Four gates:
  1. Forward matches Phase 4A reference within INT8 noise (cos_sim > 0.9999)
  2. Backward matches Phase 4A reference within bf16 noise
  3. torch.compile(fullgraph=True, dynamic=False) compatibility
  4. [harness smoke — run separately via W8A8_TRITON=1 harness run]

Run from repo root:
    python experiments/test_w8a8_triton_kernel.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault(
    "CC",
    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe",
)

import torch
import torch.nn.functional as F

from fused_kernels import (
    w8a8_linear_reference,
    w8a8_linear_triton,
    fused_mlp_up_w8a8_triton,
)


def gate_1_forward_matches_reference() -> None:
    """Triton INT8 GEMM should match reference fp32 fake-quant to within noise."""
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 8, 1024, 512, 1024  # pr-1493 MLP fc shape

    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06

    with torch.no_grad():
        ref = w8a8_linear_reference(x, w)
        tri = w8a8_linear_triton(x, w)

    assert ref.shape == tri.shape, f"shape mismatch: ref {ref.shape}, tri {tri.shape}"
    assert tri.dtype == torch.bfloat16

    cos = F.cosine_similarity(ref.float().flatten(), tri.float().flatten(), dim=0).item()
    diff = (ref.float() - tri.float()).abs()
    max_diff = diff.max().item()
    max_ref = ref.float().abs().max().item()
    rel_max = max_diff / max(max_ref, 1e-30)

    # Reference and Triton both quantize to int8 with the same per-row scales
    # (computed via the same formula). The Triton path does int32 accumulate
    # (exact) + fp32 scale multiply. Reference does fp32 matmul on dequantized
    # values. The results should be essentially identical modulo the order of
    # (int32 accumulate) vs (fp32 accumulate of dequantized values).
    if cos < 0.9999:
        raise AssertionError(f"Gate 1 FAIL: forward cos {cos:.6f} < 0.9999")
    if rel_max > 5e-3:
        raise AssertionError(
            f"Gate 1 FAIL: forward rel_max {rel_max:.3e} > 5e-3 "
            f"(max_diff {max_diff:.3e}, max_ref {max_ref:.3e})"
        )
    print(
        f"[PASS] Gate 1 forward: cos {cos:.6f}, rel_max {rel_max:.2e} "
        f"(max_diff {max_diff:.2e}, max_ref {max_ref:.2e})"
    )


def gate_2_backward_matches_reference() -> None:
    """STE backward via register_autograd should match reference STE."""
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 4, 512, 512, 1024

    x_a = torch.randn(B, T, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    w_a = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06
    w_a.requires_grad_(True)

    x_b = x_a.detach().clone().requires_grad_(True)
    w_b = w_a.detach().clone().requires_grad_(True)

    ref_out = w8a8_linear_reference(x_a, w_a)
    ref_out.float().pow(2).mean().backward()

    tri_out = w8a8_linear_triton(x_b, w_b)
    tri_out.float().pow(2).mean().backward()

    assert x_a.grad is not None and x_b.grad is not None
    assert w_a.grad is not None and w_b.grad is not None

    gx_cos = F.cosine_similarity(x_a.grad.float().flatten(), x_b.grad.float().flatten(), dim=0).item()
    gw_cos = F.cosine_similarity(w_a.grad.flatten(), w_b.grad.flatten(), dim=0).item()
    gx_max = (x_a.grad.float() - x_b.grad.float()).abs().max().item()
    gw_max = (w_a.grad - w_b.grad).abs().max().item()

    if gx_cos < 0.9999:
        raise AssertionError(f"Gate 2 FAIL: grad_x cos {gx_cos:.6f} < 0.9999")
    if gw_cos < 0.9999:
        raise AssertionError(f"Gate 2 FAIL: grad_w cos {gw_cos:.6f} < 0.9999")

    print(
        f"[PASS] Gate 2 backward: grad_x cos {gx_cos:.6f} (max {gx_max:.2e}), "
        f"grad_w cos {gw_cos:.6f} (max {gw_max:.2e})"
    )


def gate_3_compile_safe() -> None:
    """W8A8 Triton block must compile under fullgraph=True."""
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 2, 512, 512, 1024

    class W8A8TritonBlock(torch.nn.Module):
        def __init__(self, K: int, N: int) -> None:
            super().__init__()
            self.fc_w = torch.nn.Parameter(torch.randn(N, K, dtype=torch.float32) * 0.06)
            self.proj_w = torch.nn.Parameter(torch.randn(K, N, dtype=torch.float32) * 0.06)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = fused_mlp_up_w8a8_triton(x, self.fc_w)
            return w8a8_linear_triton(h, self.proj_w)

    m = W8A8TritonBlock(K, N).to(device)
    try:
        compiled = torch.compile(m, fullgraph=True, dynamic=False)
    except Exception as e:
        raise AssertionError(f"Gate 3 FAIL (compile call): {e}") from e

    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    try:
        out = compiled(x)
    except Exception as e:
        raise AssertionError(f"Gate 3 FAIL (compiled forward): {e}") from e

    assert out.shape == (B, T, K), f"unexpected shape {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in compiled forward output"

    try:
        out.float().sum().backward()
    except Exception as e:
        raise AssertionError(f"Gate 3 FAIL (compiled backward): {e}") from e

    assert m.fc_w.grad is not None and torch.isfinite(m.fc_w.grad).all()
    assert m.proj_w.grad is not None and torch.isfinite(m.proj_w.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[PASS] Gate 3 compile: fullgraph=True fwd+bwd on Triton W8A8 block succeeded")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gate_1_forward_matches_reference()
    gate_2_backward_matches_reference()
    gate_3_compile_safe()
    print("\n[ALL GATES PASS] Triton W8A8 kernel matches reference and is compile-safe")


if __name__ == "__main__":
    main()
