"""Validation gates for W8A8 pure-PyTorch reference implementation.

Phase 4A — verify the reference path is numerically reasonable and compile-safe
BEFORE writing a Triton INT8-cores kernel. If any gate fails, quantization is
not compatible with pr-1493's dynamics and a Triton kernel cannot fix that.

Four gates:
  1. Quantization roundtrip sanity (cos_sim > 0.995, no NaN)
  2. STE gradient flow (matches manual "matmul on quantized, STE backward")
  3. Forward error magnitude (cos_sim(W8A8, fp32) > 0.98)
  4. torch.compile(fullgraph=True, dynamic=False) compatibility

Run from repo root:
    python experiments/test_w8a8_reference.py
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
    _quantize_to_int8_per_row,
    _ste_fake_quant,
    w8a8_linear_reference,
    fused_mlp_up_w8a8_reference,
)


def gate_1_quantization_roundtrip() -> None:
    torch.manual_seed(1337)
    device = torch.device("cuda")

    # Sweep tensor scales — quantization must handle typical activation
    # magnitudes (~1) and typical weight magnitudes (~0.05 at Kaiming init).
    for scale in [1.0, 0.05, 10.0, 1e-3]:
        t = torch.randn(64, 512, dtype=torch.float32, device=device) * scale
        t_q = _quantize_to_int8_per_row(t)

        assert t_q.shape == t.shape
        assert not torch.isnan(t_q).any(), f"NaN at scale {scale}"
        assert not torch.isinf(t_q).any(), f"Inf at scale {scale}"

        cos = F.cosine_similarity(t.flatten(), t_q.flatten(), dim=0).item()
        # For symmetric per-row INT8, cos > 0.9995 is routine for typical tensors.
        if cos < 0.995:
            raise AssertionError(f"Gate 1 FAIL at scale {scale}: cos {cos:.6f}")

        # Verify per-row quantized values are exactly N*scale for some integer N
        # in [-127, 127] (sanity: quant is exact, not just approximate).
        amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        per_row_scale = amax / 127.0
        ratios = t_q / per_row_scale
        int_err = (ratios - ratios.round()).abs().max().item()
        if int_err > 1e-4:
            raise AssertionError(
                f"Gate 1 FAIL at scale {scale}: quantized values not integer multiples "
                f"of scale (err {int_err:.2e})"
            )

    print("[PASS] Gate 1 quantization roundtrip: cos>0.995 across 4 magnitude regimes")


def gate_2_ste_gradient_flow() -> None:
    """STE semantics: gradient to unquantized inputs equals the gradient
    one would compute manually using quantized values and treating quant
    as identity in backward.

    Math:
        forward: out = F.linear(q(x), q(w))  where q = fake_quant
        STE:     d(q(t))/dt = 1   (identity)
        So:      dL/dx = grad_out @ q(w)
                 dL/dw = grad_out^T @ q(x)
    """
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 4, 128, 512, 1024

    x = torch.randn(B, T, K, dtype=torch.float32, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06
    w.requires_grad_(True)

    # Autograd path
    out = w8a8_linear_reference(x, w)
    # A simple scalar loss so we have a deterministic grad_out
    loss = out.sum()
    loss.backward()
    grad_x_ste = x.grad.clone()
    grad_w_ste = w.grad.clone()

    # Manual reference computation
    with torch.no_grad():
        x_q = _quantize_to_int8_per_row(x.detach())
        w_q = _quantize_to_int8_per_row(w.detach())
        grad_out = torch.ones(B, T, N, dtype=torch.float32, device=device)
        # dL/dx = grad_out @ w_q     (F.linear: out = x @ w.T, d out/d x = w)
        grad_x_expected = grad_out @ w_q  # [B, T, N] @ [N, K] -> [B, T, K]
        # dL/dw = grad_out.T @ x_q   (out = x @ w.T, d out/d w = x; accumulate over batch)
        grad_out_flat = grad_out.reshape(-1, N)
        x_q_flat = x_q.reshape(-1, K)
        grad_w_expected = grad_out_flat.t() @ x_q_flat  # [N, M] @ [M, K] -> [N, K]

    gx_cos = F.cosine_similarity(grad_x_ste.flatten(), grad_x_expected.flatten(), dim=0).item()
    gw_cos = F.cosine_similarity(grad_w_ste.flatten(), grad_w_expected.flatten(), dim=0).item()
    gx_max = (grad_x_ste - grad_x_expected).abs().max().item()
    gw_max = (grad_w_ste - grad_w_expected).abs().max().item()

    if gx_cos < 0.9999:
        raise AssertionError(f"Gate 2 FAIL: grad_x cos {gx_cos:.6f} < 0.9999")
    if gw_cos < 0.9999:
        raise AssertionError(f"Gate 2 FAIL: grad_w cos {gw_cos:.6f} < 0.9999")

    print(
        f"[PASS] Gate 2 STE gradient flow: grad_x cos {gx_cos:.6f} (max {gx_max:.2e}), "
        f"grad_w cos {gw_cos:.6f} (max {gw_max:.2e})"
    )


def gate_3_forward_quant_noise() -> None:
    """Quantization noise on the forward output should be small enough for
    training to be viable — cos_sim > 0.98 vs fp32 reference, mean rel err < 5%.
    """
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 4, 256, 512, 1024

    x = torch.randn(B, T, K, dtype=torch.float32, device=device)
    w = torch.randn(N, K, dtype=torch.float32, device=device) * 0.06

    with torch.no_grad():
        out_fp32 = F.linear(x, w)
        out_w8a8 = w8a8_linear_reference(x, w)

    cos = F.cosine_similarity(out_fp32.flatten(), out_w8a8.flatten(), dim=0).item()
    diff = (out_fp32 - out_w8a8).abs()
    rel_mean = diff.mean().item() / out_fp32.abs().mean().clamp_min(1e-30).item()

    if cos < 0.98:
        raise AssertionError(f"Gate 3 FAIL: forward cos {cos:.6f} < 0.98")
    if rel_mean > 0.05:
        raise AssertionError(f"Gate 3 FAIL: mean rel err {rel_mean:.3%} > 5%")

    print(f"[PASS] Gate 3 forward quant noise: cos {cos:.6f}, mean rel err {rel_mean:.2%}")


def gate_4_compile_safe() -> None:
    torch.manual_seed(1337)
    device = torch.device("cuda")
    B, T, K, N = 2, 512, 512, 1024

    class W8A8Block(torch.nn.Module):
        def __init__(self, K: int, N: int) -> None:
            super().__init__()
            self.fc_w = torch.nn.Parameter(torch.randn(N, K, dtype=torch.float32) * 0.06)
            self.proj_w = torch.nn.Parameter(torch.randn(K, N, dtype=torch.float32) * 0.06)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = fused_mlp_up_w8a8_reference(x, self.fc_w)
            return w8a8_linear_reference(h, self.proj_w)

    m = W8A8Block(K, N).to(device)

    try:
        compiled = torch.compile(m, fullgraph=True, dynamic=False)
    except Exception as e:
        raise AssertionError(f"Gate 4 FAIL (compile call): {e}") from e

    x = torch.randn(B, T, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    try:
        out = compiled(x)
    except Exception as e:
        raise AssertionError(f"Gate 4 FAIL (compiled forward): {e}") from e

    assert out.shape == (B, T, K), f"shape {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in compiled forward output"

    try:
        out.float().sum().backward()
    except Exception as e:
        raise AssertionError(f"Gate 4 FAIL (compiled backward): {e}") from e

    assert m.fc_w.grad is not None and torch.isfinite(m.fc_w.grad).all()
    assert m.proj_w.grad is not None and torch.isfinite(m.proj_w.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[PASS] Gate 4 compile: fullgraph=True fwd+bwd on W8A8 block succeeded")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    gate_1_quantization_roundtrip()
    gate_2_ste_gradient_flow()
    gate_3_forward_quant_noise()
    gate_4_compile_safe()
    print("\n[ALL GATES PASS] W8A8 reference is correct, STE-sound, compile-safe")


if __name__ == "__main__":
    main()
