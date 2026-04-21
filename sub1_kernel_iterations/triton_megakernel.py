"""
Full Megakernel: Fuse through the matmuls so the [M, 1536] intermediate never hits HBM.

Key idea: tile the hidden dimension H=1536 into chunks of BLOCK_K. For each chunk:
  1. Compute partial up-projection [BLOCK_M, BLOCK_K] in SRAM
  2. Apply activation in registers
  3. Accumulate partial down-projection [BLOCK_M, D] in registers

The [M, 1536] intermediate stays entirely in SRAM/registers.

Shared memory budget (RTX 5070 Ti): 101 KB
With BLOCK_M=8, BLOCK_K=32, D=512:
  - x tile:        8 × 512 × 2 = 8 KB (bf16)
  - up_w chunk:   32 × 512 × 2 = 32 KB (bf16, loaded per tile)
  - down_w chunk: 512 × 32 × 2 = 32 KB (bf16, loaded per tile)
  - hidden chunk:  8 × 32 × 4 = 1 KB (f32, registers)
  - out_acc:       8 × 512 × 4 = 16 KB (f32, registers)
  - x_residual:    8 × 512 × 4 = 16 KB (f32, registers)
  Total: ~105 KB — tight but may work with num_stages=1 and careful management
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
import time


# We can't use D=512 as constexpr because it's too large for SRAM.
# Instead, we process D in tiles too (double tiling).
# Actually — let's try with D constexpr but tiny BLOCK_M.

@triton.jit
def megakernel_mlp_fwd(
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
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    d_offs = tl.arange(0, D)

    # Load x [BLOCK_M, D] — single HBM read
    x_ptrs = X_ptr + m_offs[:, None] * stride_x_m + d_offs[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    x_residual = x_tile

    # RMSNorm + ln_scale
    var = tl.sum(x_tile * x_tile, axis=1) / D
    x_normed = x_tile * tl.rsqrt(var[:, None] + 1e-6) * LN_SCALE

    # Accumulate down-projection output [BLOCK_M, D]
    out_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Tile over hidden dimension H
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        # Load up_w chunk [BLOCK_K, D]
        uw_ptrs = UP_W_ptr + k_offs[:, None] * stride_uw_h + d_offs[None, :] * stride_uw_d
        up_w = tl.load(uw_ptrs, mask=k_mask[:, None], other=0.0)

        # Up projection: [BLOCK_M, D] @ [D, BLOCK_K] = [BLOCK_M, BLOCK_K]
        hidden = tl.dot(x_normed.to(tl.bfloat16), tl.trans(up_w).to(tl.bfloat16)).to(tl.float32)

        # LeakyReLU(0.5)² in registers
        hidden = tl.where(hidden > 0, hidden, 0.5 * hidden)
        hidden = hidden * hidden

        # Load down_w chunk [D, BLOCK_K]
        dw_ptrs = DOWN_W_ptr + d_offs[:, None] * stride_dw_d + k_offs[None, :] * stride_dw_h
        down_w = tl.load(dw_ptrs, mask=k_mask[None, :], other=0.0)

        # Down projection: [BLOCK_M, BLOCK_K] @ [BLOCK_K, D] = [BLOCK_M, D]
        out_acc += tl.dot(hidden.to(tl.bfloat16), tl.trans(down_w).to(tl.bfloat16)).to(tl.float32)

    # Scale + residual
    mlp_scale = tl.load(MLP_SCALE_ptr + d_offs).to(tl.float32)
    result = x_residual + mlp_scale[None, :] * out_acc

    # Single HBM write
    out_ptrs = OUT_ptr + m_offs[:, None] * stride_o_m + d_offs[None, :] * stride_o_d
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=m_mask[:, None])


def megakernel_mlp_forward(x, up_w, down_w, mlp_scale, ln_scale, block_m=8, block_k=32):
    """Forward pass using the megakernel."""
    B, S, D = x.shape
    H = up_w.shape[0]
    M = B * S

    x_flat = x.reshape(M, D).contiguous().to(torch.bfloat16)
    uw = up_w.contiguous().to(torch.bfloat16)
    dw = down_w.contiguous().to(torch.bfloat16)
    ms = mlp_scale.contiguous().float()
    out = torch.empty_like(x_flat)

    grid = (triton.cdiv(M, block_m),)
    megakernel_mlp_fwd[grid](
        x_flat, uw, dw, ms, out,
        M, D, H, ln_scale,
        x_flat.stride(0), x_flat.stride(1),
        uw.stride(0), uw.stride(1),
        dw.stride(0), dw.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_K=block_k,
    )
    return out.reshape(B, S, D)


def reference_mlp(x, up_w, down_w, mlp_scale, ln_scale):
    """Reference PyTorch implementation."""
    x_normed = F.rms_norm(x, (x.size(-1),)) * ln_scale
    hidden = F.leaky_relu(F.linear(x_normed, up_w.to(x.dtype)), negative_slope=0.5)
    mlp_out = F.linear(hidden.square(), down_w.to(x.dtype))
    return x + mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out


def test_megakernel():
    torch.manual_seed(42)
    B, S, D, H = 2, 64, 512, 1536
    ln_scale = 1.0 / 3.0 ** 0.5

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32)

    # Try different block sizes
    for bm, bk in [(8, 32), (4, 32), (8, 16), (4, 16)]:
        try:
            out = megakernel_mlp_forward(x, up_w, down_w, mlp_scale, ln_scale, block_m=bm, block_k=bk)
            ref = reference_mlp(x, up_w, down_w, mlp_scale, ln_scale)
            cos = F.cosine_similarity(out.float().reshape(1, -1), ref.float().reshape(1, -1)).item()
            abs_err = (out.float() - ref.float()).abs().max().item()
            print(f"BLOCK_M={bm:2d}, BLOCK_K={bk:2d}: cos_sim={cos:.8f}, abs_err={abs_err:.1f}")
        except Exception as e:
            print(f"BLOCK_M={bm:2d}, BLOCK_K={bk:2d}: FAILED — {e}")


def benchmark_megakernel():
    torch.manual_seed(42)
    B, S, D, H = 2, 2048, 512, 1536  # Realistic
    ln_scale = 1.0 / 3.0 ** 0.5

    x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16)
    up_w = torch.randn(H, D, device='cuda', dtype=torch.float32)
    down_w = torch.randn(D, H, device='cuda', dtype=torch.float32)
    mlp_scale = torch.ones(D, device='cuda', dtype=torch.float32)

    # Find working block sizes
    working_config = None
    for bm, bk in [(8, 32), (4, 32), (8, 16), (4, 16)]:
        try:
            megakernel_mlp_forward(x, up_w, down_w, mlp_scale, ln_scale, block_m=bm, block_k=bk)
            working_config = (bm, bk)
            break
        except:
            continue

    if working_config is None:
        print("No working config found!")
        return

    bm, bk = working_config
    print(f"\nBenchmark with BLOCK_M={bm}, BLOCK_K={bk}, B={B}, S={S}")

    # Warmup
    for _ in range(20):
        reference_mlp(x, up_w, down_w, mlp_scale, ln_scale)
    for _ in range(20):
        megakernel_mlp_forward(x, up_w, down_w, mlp_scale, ln_scale, bm, bk)

    torch.cuda.synchronize()

    # Benchmark reference (forward only)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        reference_mlp(x, up_w, down_w, mlp_scale, ln_scale)
    end.record()
    torch.cuda.synchronize()
    ref_ms = start.elapsed_time(end) / 100

    # Benchmark megakernel (forward only)
    start.record()
    for _ in range(100):
        megakernel_mlp_forward(x, up_w, down_w, mlp_scale, ln_scale, bm, bk)
    end.record()
    torch.cuda.synchronize()
    mega_ms = start.elapsed_time(end) / 100

    print(f"  Reference (PyTorch): {ref_ms:.3f} ms")
    print(f"  Megakernel (Triton): {mega_ms:.3f} ms")
    print(f"  Speedup: {ref_ms/mega_ms:.2f}x ({(1-mega_ms/ref_ms)*100:+.1f}%)")


if __name__ == "__main__":
    print("=== Testing megakernel correctness ===")
    test_megakernel()
    print("\n=== Benchmarking megakernel ===")
    benchmark_megakernel()
