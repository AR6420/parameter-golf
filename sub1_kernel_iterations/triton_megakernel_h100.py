"""
H100-optimized Megakernel for Parameter Golf.

Target: NVIDIA H100 SXM (sm_90, 228 KB shared memory per SM)

Fuses the entire MLP path into a single kernel:
  RMSNorm → LN_scale → UpProj → LeakyReLU(0.5)² → DownProj → MLP_scale → Residual

The [M, 1536] MLP intermediate NEVER touches HBM — stays in SRAM/registers.

SRAM Budget (H100, 228 KB):
  BLOCK_M=16, BLOCK_K=32:
    - x tile:        16 × 512 × 2 = 16 KB (bf16)
    - up_w chunk:    32 × 512 × 2 = 32 KB (bf16, loaded per tile)
    - down_w chunk: 512 × 32 × 2 = 32 KB (bf16, loaded per tile)
    - hidden chunk:  16 × 32 × 4 = 2 KB (f32, registers)
    - out_acc:       16 × 512 × 4 = 32 KB (f32, registers)
    - x_residual:    16 × 512 × 4 = 32 KB (f32, registers)
    Total: ~146 KB (fits in 228 KB)

  BLOCK_M=16, BLOCK_K=64:
    - up_w chunk:    64 × 512 × 2 = 64 KB
    - down_w chunk: 512 × 64 × 2 = 64 KB
    Total: ~210 KB (tight fit, may need num_stages=1)

Performance analysis (8×H100, batch_tokens=786432, grad_accum=1):
  - Per-GPU tokens: 98304
  - Intermediate avoided: 98304 × 1536 × 2 × 2 (read+write) = 576 MB per layer × 11 = 6.3 GB
  - At 3.35 TB/s bandwidth: ~1.88 ms saved
  - Element-wise ops saved: ~2.6 ms
  - Total expected savings: ~4.5 ms / 86.7 ms = ~5.2%
  - Extra steps in 600s: ~360 more steps → ~0.0002 BPB improvement

This is incremental but real. Combined with attention pre/post fusion, could reach 8-10%.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl


# ============================================================
# H100 Megakernel: Fused MLP Forward
# ============================================================

@triton.autotune(
    configs=[
        # H100 configs: 228 KB shared memory allows larger tiles
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=1),
    ],
    key=['M', 'D', 'H'],
)
@triton.jit
def fused_mlp_megakernel_fwd(
    X_ptr, UP_W_ptr, DOWN_W_ptr, MLP_SCALE_ptr, OUT_ptr,
    M, D: tl.constexpr, H,
    LN_SCALE,
    stride_x_m, stride_x_d,
    stride_uw_h, stride_uw_d,
    stride_dw_d, stride_dw_h,
    stride_o_m, stride_o_d,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Full fused MLP megakernel.
    Each program processes BLOCK_M tokens through the complete MLP.

    Data flow:
      HBM read x → [SRAM: RMSNorm → scale → tiled(UpProj → Act → DownProj) → scale + residual] → HBM write
    """
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    d_offs = tl.arange(0, D)

    # === Load x from HBM [BLOCK_M, D] — SINGLE READ ===
    x_ptrs = X_ptr + m_offs[:, None] * stride_x_m + d_offs[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x_residual = x_tile

    # === RMSNorm + LN_scale (all in registers) ===
    var = tl.sum(x_tile * x_tile, axis=1) / D
    x_normed = x_tile * tl.rsqrt(var[:, None] + 1e-6) * LN_SCALE

    # === Tiled MLP: Up → Act → Down (intermediate stays in SRAM) ===
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        # Load up_w chunk [BLOCK_K, D] from HBM
        uw_ptrs = UP_W_ptr + k_offs[:, None] * stride_uw_h + d_offs[None, :] * stride_uw_d
        up_w = tl.load(uw_ptrs, mask=k_mask[:, None], other=0.0)

        # Up projection: [BLOCK_M, D] @ [D, BLOCK_K] = [BLOCK_M, BLOCK_K]
        hidden = tl.dot(x_normed.to(tl.bfloat16), tl.trans(up_w).to(tl.bfloat16)).to(tl.float32)

        # LeakyReLU(0.5)² — entirely in registers!
        hidden = tl.where(hidden > 0, hidden, 0.5 * hidden)
        hidden = hidden * hidden

        # Load down_w chunk [D, BLOCK_K] from HBM
        dw_ptrs = DOWN_W_ptr + d_offs[:, None] * stride_dw_d + k_offs[None, :] * stride_dw_h
        down_w = tl.load(dw_ptrs, mask=k_mask[None, :], other=0.0)

        # Down projection: [BLOCK_M, BLOCK_K] @ [BLOCK_K, D] = [BLOCK_M, D]
        out_acc += tl.dot(hidden.to(tl.bfloat16), tl.trans(down_w).to(tl.bfloat16)).to(tl.float32)

    # === Scale + Residual (in registers) ===
    mlp_scale = tl.load(MLP_SCALE_ptr + d_offs).to(tl.float32)
    result = x_residual + mlp_scale[None, :] * out_acc

    # === Write result to HBM — SINGLE WRITE ===
    out_ptrs = OUT_ptr + m_offs[:, None] * stride_o_m + d_offs[None, :] * stride_o_d
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=m_mask[:, None])


# ============================================================
# autograd Function with standard PyTorch backward
# ============================================================

class FusedMLPMegakernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_w, down_w, mlp_scale, ln_scale_factor):
        B, S, D = x.shape
        H = up_w.shape[0]
        M = B * S

        x_flat = x.reshape(M, D).contiguous().to(torch.bfloat16)
        uw = up_w.contiguous().to(torch.bfloat16)
        dw = down_w.contiguous().to(torch.bfloat16)
        ms = mlp_scale.contiguous().float()
        out = torch.empty_like(x_flat)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_mlp_megakernel_fwd[grid](
            x_flat, uw, dw, ms, out,
            M, D, H, ln_scale_factor,
            x_flat.stride(0), x_flat.stride(1),
            uw.stride(0), uw.stride(1),
            dw.stride(0), dw.stride(1),
            out.stride(0), out.stride(1),
        )

        # Save for backward (standard recomputation)
        ctx.save_for_backward(x, up_w, down_w, mlp_scale)
        ctx.ln_scale_factor = ln_scale_factor
        return out.reshape(B, S, D)

    @staticmethod
    def backward(ctx, grad_output):
        x, up_w, down_w, mlp_scale = ctx.saved_tensors
        ln_scale = ctx.ln_scale_factor
        B, S, D = x.shape
        M = B * S

        x_flat = x.reshape(M, D).float()

        # Recompute forward intermediates
        variance = (x_flat ** 2).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + 1e-6)
        x_normed = x_flat * rms_inv * ln_scale

        hidden = F.linear(x_normed.to(up_w.dtype), up_w)
        leaky = torch.where(hidden.float() > 0, hidden.float(), 0.5 * hidden.float())
        activated = leaky ** 2
        mlp_out = F.linear(activated.to(down_w.dtype), down_w)

        grad_flat = grad_output.reshape(M, D).float()

        # Backward: d(x + mlp_scale * mlp_out)
        grad_mlp_scale = (grad_flat * mlp_out.float()).sum(dim=0)
        grad_scaled = grad_flat * mlp_scale.float().unsqueeze(0)

        # d(down proj)
        grad_activated = F.linear(grad_scaled.to(down_w.dtype), down_w.t())
        grad_down_w = grad_scaled.t().to(activated.dtype) @ activated.to(grad_scaled.dtype)

        # d(activation)
        leaky_coeff = torch.where(hidden.float() > 0, 1.0, 0.5)
        grad_hidden = grad_activated.float() * 2.0 * leaky * leaky_coeff

        # d(up proj)
        grad_x_normed = F.linear(grad_hidden.to(up_w.dtype), up_w.t())
        grad_up_w = grad_hidden.t().to(x_normed.dtype) @ x_normed.to(grad_hidden.dtype)

        # d(RMSNorm * ln_scale)
        grad_xn = grad_x_normed.float() * ln_scale
        d_x = grad_xn * rms_inv
        d_x -= x_flat * (grad_xn * x_flat).sum(dim=-1, keepdim=True) * (rms_inv ** 3) / D

        grad_x = grad_flat + d_x

        return (
            grad_x.to(x.dtype).reshape(B, S, D),
            grad_up_w.to(up_w.dtype),
            grad_down_w.to(down_w.dtype),
            grad_mlp_scale,
            None,
        )


def fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale_factor, use_triton=True):
    """Drop-in replacement for the MLP path in Block.forward().
    Uses the megakernel on CUDA, falls back to PyTorch otherwise."""
    if use_triton and x.is_cuda:
        return FusedMLPMegakernelFunction.apply(x, up_w, down_w, mlp_scale, ln_scale_factor)

    # Fallback
    x_normed = F.rms_norm(x, (x.size(-1),)) * ln_scale_factor
    hidden = F.leaky_relu(F.linear(x_normed, up_w.to(x.dtype)), negative_slope=0.5)
    mlp_out = F.linear(hidden.square(), down_w.to(x.dtype))
    return x + mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out


# ============================================================
# Tests (will work on H100, OOM on smaller GPUs)
# ============================================================

def test_correctness():
    """Test on whatever GPU is available."""
    import os
    torch.manual_seed(42)
    B, S, D, H = 2, 64, 512, 1536
    ln_scale = 1.0 / 3.0 ** 0.5

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32)

    ref = fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=False)

    try:
        out = fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=True)
        cos = F.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
        abs_err = (out.float() - ref.float()).abs().max().item()
        print(f"Megakernel cos_sim={cos:.8f}, abs_err={abs_err:.1f}")
        assert cos > 0.999, f"Cosine similarity too low: {cos}"
        print("PASSED!")
    except RuntimeError as e:
        if "shared memory" in str(e) or "out of resource" in str(e):
            print(f"Skipped: GPU shared memory too small ({e})")
            print("This kernel requires H100 (228 KB shared memory)")
        else:
            raise


def benchmark():
    torch.manual_seed(42)
    B, S, D, H = 48, 2048, 512, 1536  # H100-scale batch
    ln_scale = 1.0 / 3.0 ** 0.5

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32)

    try:
        # Warmup
        for _ in range(5):
            fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=True)
        for _ in range(5):
            fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=False)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Megakernel
        start.record()
        for _ in range(50):
            fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=True)
        end.record()
        torch.cuda.synchronize()
        mega_ms = start.elapsed_time(end) / 50

        # Reference
        start.record()
        for _ in range(50):
            fused_mlp_megakernel(x, up_w, down_w, mlp_scale, ln_scale, use_triton=False)
        end.record()
        torch.cuda.synchronize()
        ref_ms = start.elapsed_time(end) / 50

        print(f"\nH100-scale benchmark (B={B}, S={S}):")
        print(f"  Reference: {ref_ms:.3f} ms")
        print(f"  Megakernel: {mega_ms:.3f} ms")
        print(f"  Speedup: {ref_ms/mega_ms:.2f}x ({(1-mega_ms/ref_ms)*100:+.1f}%)")

    except RuntimeError as e:
        if "shared memory" in str(e) or "out of resource" in str(e):
            print(f"Benchmark skipped: GPU too small. Test on H100.")
        elif "out of memory" in str(e):
            print(f"Benchmark skipped: Not enough VRAM for H100-scale batch. Test on H100.")
        else:
            raise


if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()
    print("\n=== Benchmark ===")
    benchmark()
