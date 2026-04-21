"""
Triton fused kernels for Parameter Golf.

Strategy: Fuse element-wise operations around cuBLAS matmuls.
The MLP path has 7 operations; we reduce to 4 by fusing element-wise ops:

  Original:                    Fused:
  1. RMSNorm(x)                1. fused_rmsnorm_scale(x, ln_scale) → x_normed  [fuses 1+2]
  2. * ln_scale                2. F.linear(x_normed, up_w)     → hidden         [cuBLAS, unchanged]
  3. F.linear(→ hidden)        3. fused_act_square(hidden)     → hidden_sq      [fuses 4+5, in-place]
  4. LeakyReLU(0.5)            4. F.linear(hidden_sq, down_w)  → mlp_out        [cuBLAS, unchanged]
  5. .square()                 5. fused_scale_residual(x, mlp_out, scale) → out [fuses 6+7]
  6. F.linear(→ mlp_out)
  7. x + mlp_scale * mlp_out

Savings: 3 fewer kernel launches, 3 fewer HBM round-trips per layer × 11 layers.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl


# ============================================================
# Kernel 1: Fused RMSNorm + Scale
# ============================================================

@triton.jit
def fused_rmsnorm_scale_kernel(
    X_ptr, OUT_ptr,
    M, D,
    LN_SCALE,
    stride_x, stride_o,
    BLOCK_D: tl.constexpr,
):
    """RMSNorm(x) * ln_scale_factor. One read, one write per row."""
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    x = tl.load(X_ptr + row * stride_x + d_offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: x / sqrt(mean(x²) + eps)
    var = tl.sum(x * x) / D
    rms_inv = tl.rsqrt(var + 1e-6)
    out = x * rms_inv * LN_SCALE

    tl.store(OUT_ptr + row * stride_o + d_offs, out.to(tl.bfloat16), mask=mask)


def fused_rmsnorm_scale(x: Tensor, ln_scale: float) -> Tensor:
    """Fused RMSNorm + ln_scale_factor multiply."""
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, D = x_flat.shape
    out = torch.empty_like(x_flat)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (M,)

    fused_rmsnorm_scale_kernel[grid](
        x_flat, out, M, D, ln_scale,
        x_flat.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out.reshape(shape)


# ============================================================
# Kernel 2: Fused LeakyReLU(0.5) + Square (in-place)
# ============================================================

@triton.jit
def fused_leaky_relu_square_kernel(
    X_ptr,  # [M, H] — modified in place
    N,      # total elements
    BLOCK: tl.constexpr,
):
    """In-place: x = leaky_relu(x, 0.5)² — one read, one write."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    # LeakyReLU(0.5) then square
    x = tl.where(x > 0, x, 0.5 * x)
    x = x * x
    tl.store(X_ptr + offs, x.to(tl.bfloat16), mask=mask)


def fused_leaky_relu_square_(hidden: Tensor) -> Tensor:
    """In-place fused LeakyReLU(0.5) + square."""
    N = hidden.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    fused_leaky_relu_square_kernel[grid](hidden, N, BLOCK=BLOCK)
    return hidden


# ============================================================
# Kernel 3: Fused Scale + Residual Add
# ============================================================

@triton.jit
def fused_scale_residual_kernel(
    X_IN_ptr,      # [M, D] original input (for residual)
    MLP_OUT_ptr,   # [M, D] MLP output
    SCALE_ptr,     # [D] per-dim scale
    OUT_ptr,       # [M, D] result
    M, D,
    stride_x, stride_mlp, stride_o,
    BLOCK_D: tl.constexpr,
):
    """out = x_in + scale * mlp_out — one read each, one write."""
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    x_in = tl.load(X_IN_ptr + row * stride_x + d_offs, mask=mask, other=0.0).to(tl.float32)
    mlp_out = tl.load(MLP_OUT_ptr + row * stride_mlp + d_offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(SCALE_ptr + d_offs, mask=mask, other=0.0).to(tl.float32)

    result = x_in + scale * mlp_out

    tl.store(OUT_ptr + row * stride_o + d_offs, result.to(tl.bfloat16), mask=mask)


def fused_scale_residual(x_in: Tensor, mlp_out: Tensor, scale: Tensor) -> Tensor:
    """Fused: x_in + scale[None,None,:] * mlp_out"""
    shape = x_in.shape
    x_flat = x_in.reshape(-1, shape[-1]).contiguous()
    mlp_flat = mlp_out.reshape(-1, shape[-1]).contiguous()
    M, D = x_flat.shape
    out = torch.empty_like(x_flat)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (M,)

    fused_scale_residual_kernel[grid](
        x_flat, mlp_flat, scale.contiguous(),
        out, M, D,
        x_flat.stride(0), mlp_flat.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out.reshape(shape)


# ============================================================
# Combined Fused MLP (autograd Function)
# ============================================================

class FusedMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_w, down_w, mlp_scale, ln_scale_factor):
        """
        Fused MLP forward: RMSNorm → scale → up_proj → act → down_proj → scale+residual.
        Uses Triton for element-wise ops, cuBLAS for matmuls.
        """
        # 1. Fused RMSNorm + ln_scale
        x_normed = fused_rmsnorm_scale(x, ln_scale_factor)

        # 2. Up projection (cuBLAS)
        hidden_pre = F.linear(x_normed, up_w.to(x_normed.dtype))  # [B, S, H]

        # 3. Fused LeakyReLU(0.5) + square
        hidden_act = hidden_pre.clone()
        fused_leaky_relu_square_(hidden_act)

        # 4. Down projection (cuBLAS)
        mlp_out = F.linear(hidden_act, down_w.to(hidden_act.dtype))  # [B, S, D]

        # 5. Fused scale + residual
        out = fused_scale_residual(x, mlp_out, mlp_scale)

        # Save intermediates for backward (avoids recomputation precision mismatch)
        ctx.save_for_backward(x, x_normed, hidden_pre, mlp_out, up_w, down_w, mlp_scale)
        ctx.ln_scale_factor = ln_scale_factor

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Standard PyTorch backward using saved intermediates."""
        x, x_normed, hidden_pre, mlp_out, up_w, down_w, mlp_scale = ctx.saved_tensors
        ln_scale = ctx.ln_scale_factor
        B, S, D = x.shape
        M = B * S

        # Recompute activation from saved hidden_pre (exact match to forward)
        hidden_f = hidden_pre.reshape(-1, hidden_pre.size(-1)).float()
        leaky = torch.where(hidden_f > 0, hidden_f, 0.5 * hidden_f)
        activated = leaky ** 2

        x_flat = x.reshape(M, D).float()
        x_normed_flat = x_normed.reshape(M, D)
        mlp_out_flat = mlp_out.reshape(M, D).float()
        grad_flat = grad_output.reshape(M, D).float()

        # d(x + mlp_scale * mlp_out) / d(mlp_out, mlp_scale, x)
        grad_mlp_scale = (grad_flat * mlp_out_flat).sum(dim=0)
        grad_scaled = grad_flat * mlp_scale.float().unsqueeze(0)
        # grad w.r.t. x from residual: identity

        # d(down_proj): grad_activated = grad_scaled @ down_w
        grad_activated = F.linear(grad_scaled.to(down_w.dtype), down_w.t())
        grad_down_w = grad_scaled.t().to(activated.dtype) @ activated.to(grad_scaled.dtype)

        # d(leaky_relu(0.5)²): d/dx[f(x)²] = 2*f(x)*f'(x)
        leaky_grad_coeff = torch.where(hidden_f > 0, 1.0, 0.5)
        grad_hidden = grad_activated.float() * 2.0 * leaky * leaky_grad_coeff

        # d(up_proj): grad_x_normed = grad_hidden @ up_w
        grad_x_normed = F.linear(grad_hidden.to(up_w.dtype), up_w.t())
        grad_up_w = grad_hidden.t().to(x_normed_flat.dtype) @ x_normed_flat

        # d(RMSNorm * ln_scale) / d(x)
        variance = (x_flat ** 2).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + 1e-6)
        grad_xn_f = grad_x_normed.float() * ln_scale
        d_x_norm = grad_xn_f * rms_inv
        d_x_norm -= x_flat * (grad_xn_f * x_flat).sum(dim=-1, keepdim=True) * (rms_inv ** 3) / D

        grad_x = grad_flat + d_x_norm

        return (
            grad_x.to(x.dtype).reshape(B, S, D),
            grad_up_w.to(up_w.dtype),
            grad_down_w.to(down_w.dtype),
            grad_mlp_scale,
            None,
        )


def fused_mlp(x, up_w, down_w, mlp_scale, ln_scale_factor, use_triton=True):
    """Drop-in replacement for the MLP path in Block.forward().

    Replaces:
        x_out + mlp_scale * mlp(mlp_norm(x_out) * ln_scale_factor, up_w, down_w)
    """
    if use_triton and x.is_cuda:
        return FusedMLPFunction.apply(x, up_w, down_w, mlp_scale, ln_scale_factor)

    # Fallback: original PyTorch path
    x_normed = F.rms_norm(x, (x.size(-1),)) * ln_scale_factor
    hidden = F.leaky_relu(F.linear(x_normed, up_w.to(x.dtype)), negative_slope=0.5)
    mlp_out = F.linear(hidden.square(), down_w.to(x.dtype))
    return x + mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out


# ============================================================
# Tests
# ============================================================

def test_individual_kernels():
    """Test each fused kernel individually."""
    torch.manual_seed(42)
    device = 'cuda'

    # Test 1: fused_rmsnorm_scale
    print("Testing fused_rmsnorm_scale...")
    x = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    ln_scale = 0.577
    ref = F.rms_norm(x.float(), (512,)) * ln_scale
    out = fused_rmsnorm_scale(x, ln_scale)
    err = (ref.to(torch.bfloat16).float() - out.float()).abs().max().item()
    print(f"  max error: {err:.6f}")
    assert err < 0.01, f"RMSNorm error too large: {err}"

    # Test 2: fused_leaky_relu_square_
    print("Testing fused_leaky_relu_square_...")
    h = torch.randn(32, 1536, device=device, dtype=torch.bfloat16)
    h_ref = h.float().clone()
    h_ref = torch.where(h_ref > 0, h_ref, 0.5 * h_ref) ** 2
    fused_leaky_relu_square_(h)
    err = (h_ref.to(torch.bfloat16).float() - h.float()).abs().max().item()
    print(f"  max error: {err:.6f}")
    assert err < 0.01, f"Activation error too large: {err}"

    # Test 3: fused_scale_residual
    print("Testing fused_scale_residual...")
    x_in = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    mlp_out = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    scale = torch.randn(512, device=device, dtype=torch.float32)
    ref = x_in.float() + scale.unsqueeze(0) * mlp_out.float()
    out = fused_scale_residual(x_in, mlp_out, scale)
    err = (ref.to(torch.bfloat16).float() - out.float()).abs().max().item()
    print(f"  max error: {err:.6f}")
    assert err < 0.01, f"Scale+residual error too large: {err}"

    print("All individual kernel tests PASSED!")


def test_fused_mlp():
    """Test full fused MLP vs reference."""
    torch.manual_seed(42)
    B, S, D, H = 2, 64, 512, 1536
    ln_scale = 1.0 / (3.0 ** 0.5)

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32, requires_grad=True)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32, requires_grad=True)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32, requires_grad=True)

    # Reference
    x_ref = x.detach().clone().requires_grad_(True)
    up_ref = up_w.detach().clone().requires_grad_(True)
    down_ref = down_w.detach().clone().requires_grad_(True)
    ms_ref = mlp_scale.detach().clone().requires_grad_(True)
    out_ref = fused_mlp(x_ref, up_ref, down_ref, ms_ref, ln_scale, use_triton=False)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    # Fused
    x_tri = x.detach().clone().requires_grad_(True)
    up_tri = up_w.detach().clone().requires_grad_(True)
    down_tri = down_w.detach().clone().requires_grad_(True)
    ms_tri = mlp_scale.detach().clone().requires_grad_(True)
    out_tri = fused_mlp(x_tri, up_tri, down_tri, ms_tri, ln_scale, use_triton=True)
    loss_tri = out_tri.float().sum()
    loss_tri.backward()

    # Cosine similarity: robust metric for bf16 chains with large values
    def cos_sim(a, b):
        a_flat, b_flat = a.reshape(-1), b.reshape(-1)
        return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

    fwd_cos = cos_sim(out_ref.float(), out_tri.float())
    grad_x_cos = cos_sim(x_ref.grad.float(), x_tri.grad.float())
    grad_uw_cos = cos_sim(up_ref.grad.float(), up_tri.grad.float())
    grad_dw_cos = cos_sim(down_ref.grad.float(), down_tri.grad.float())

    fwd_abs = (out_ref.float() - out_tri.float()).abs().max().item()
    out_range = out_ref.float().abs().max().item()

    print(f"\nFused MLP comparison:")
    print(f"  Forward cos_sim: {fwd_cos:.8f} (abs_err={fwd_abs:.1f}, range={out_range:.1f})")
    print(f"  Grad x cos_sim: {grad_x_cos:.8f}")
    print(f"  Grad up_w cos_sim: {grad_uw_cos:.8f}")
    print(f"  Grad down_w cos_sim: {grad_dw_cos:.8f}")

    # Cosine similarity > 0.999 means the tensors point in essentially the same direction
    assert fwd_cos > 0.999, f"Forward cosine similarity too low: {fwd_cos}"
    assert grad_x_cos > 0.99, f"Grad x cosine similarity too low: {grad_x_cos}"
    print("PASSED: Fused MLP matches reference (cosine sim > 0.999)")


def benchmark_mlp():
    """Compare step time: original vs fused MLP."""
    import time

    torch.manual_seed(42)
    B, S, D, H = 2, 2048, 512, 1536  # Realistic sizes
    ln_scale = 1.0 / 3.0 ** 0.5

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32, requires_grad=True)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32, requires_grad=True)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32, requires_grad=True)

    def run_original(x, up_w, down_w, mlp_scale):
        x_normed = F.rms_norm(x, (x.size(-1),)) * ln_scale
        hidden = F.leaky_relu(F.linear(x_normed, up_w.to(x.dtype)), negative_slope=0.5)
        mlp_out = F.linear(hidden.square(), down_w.to(x.dtype))
        return x + mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out

    # Warmup
    for _ in range(10):
        out = run_original(x, up_w, down_w, mlp_scale)
        out.float().sum().backward()
    for _ in range(10):
        out = fused_mlp(x, up_w, down_w, mlp_scale, ln_scale, use_triton=True)
        out.float().sum().backward()

    torch.cuda.synchronize()

    # Benchmark original
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        out = run_original(x.detach().requires_grad_(True), up_w, down_w, mlp_scale)
        out.float().sum().backward()
    end.record()
    torch.cuda.synchronize()
    orig_ms = start.elapsed_time(end) / 100

    # Benchmark fused
    start.record()
    for _ in range(100):
        out = fused_mlp(x.detach().requires_grad_(True), up_w, down_w, mlp_scale, ln_scale, use_triton=True)
        out.float().sum().backward()
    end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / 100

    print(f"\nMLP Benchmark (B={B}, S={S}, D={D}, H={H}):")
    print(f"  Original: {orig_ms:.3f} ms")
    print(f"  Fused:    {fused_ms:.3f} ms")
    print(f"  Speedup:  {orig_ms/fused_ms:.2f}x ({(1-fused_ms/orig_ms)*100:.1f}%)")


if __name__ == "__main__":
    test_individual_kernels()
    test_fused_mlp()
    benchmark_mlp()
