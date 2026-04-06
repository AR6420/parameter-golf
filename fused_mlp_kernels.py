"""
Fused MLP Kernels for GEMM Boundary Fusion (Submission #2)

Kernel 1: fused_mlp_up_proj
  Computes: out = LeakyReLU(X @ W_up.T, 0.5) ** 2
  Fuses GEMM + activation + square into one kernel, eliminating
  the 1536-dim intermediate HBM round-trip.

Kernel 2: fused_mlp_down_proj  (TODO — after Kernel 1 is validated)
  Computes: out = residual + mlp_scale * (hidden @ W_down.T)

Usage:
  _USE_TRITON = True   →  fused Triton kernels
  _USE_TRITON = False  →  fallback to F.linear + manual activation
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

_USE_TRITON = True

# ─── Kernel 1: fused_mlp_up_proj ────────────────────────────────────────────
#
# C[M,N] = LeakyReLU( A[M,K] @ B[K,N], 0.5 )^2
#
# A = x (activations),  B = up_w.T  (weight transposed via strides)
# K = model_dim (512),  N = mlp_dim (1536)
# M = B*T (batch_tokens, typically 4096 per GPU on H100)
#
# Why these configs:
#   - K=512 is small → BLOCK_K=32 (16 iters) or 64 (8 iters) both fine
#   - N=1536 is 3x model_dim → BLOCK_N=128 gives 12 tiles, 256 gives 6
#   - M=4096 → BLOCK_M=64 gives 64 tiles, 128 gives 32
#   - Spike proved: 128x128x32 and 64x128x32 are competitive with cuBLAS
#   - Larger BLOCK_N (256) amortizes weight loads but needs more registers
#
# Configs ordered by expected performance (best first for early-exit):

_UP_PROJ_CONFIGS = [
    # Core configs — proven in spike (0.227ms vs cuBLAS 0.235ms)
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    # Wider-N configs — good when N >> K
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                  num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    # Smaller tiles — better for small M or high occupancy
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=5, num_warps=2),
    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4},
                  num_stages=5, num_warps=2),
]


@triton.autotune(configs=_UP_PROJ_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_mlp_up_proj_kernel(
    # Pointers
    a_ptr,      # activations  [M, K]  (contiguous)
    b_ptr,      # weight       [N, K]  (row-major, transposed via strides)
    c_ptr,      # output       [M, N]  (contiguous)
    # Dimensions
    M, N, K,
    # Strides — allow non-contiguous weight
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """GEMM + LeakyReLU(0.5) + square, fused in one kernel.

    Computes C = LeakyReLU(A @ B, 0.5)^2 where:
      A: [M, K] activations (bf16)
      B: [K, N] transposed weight (bf16, passed via strides of [N, K] tensor)
      C: [M, N] output (bf16)

    Accumulation in float32 for numerical stability.
    """
    # ── Program ID → tile coordinates (grouped for L2 locality) ──
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── Pointers for this tile ──
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ── Main GEMM loop — accumulate in float32 ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── Fused epilogue: LeakyReLU(0.5) + square ──
    # LeakyReLU: out = x if x > 0 else 0.5 * x  (in float32)
    acc = tl.where(acc > 0, acc, acc * 0.5)
    # Square (still in float32, then cast to bf16 for output)
    acc = acc * acc

    # ── Store output ──
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_mlp_up_proj(x: Tensor, weight: Tensor) -> Tensor:
    """Fused up-projection: out = LeakyReLU(x @ weight.T, 0.5)^2

    Args:
        x:      [*, K]  activations (bf16)
        weight: [N, K]  up-projection weight (bf16, same layout as nn.Linear)

    Returns:
        [*, N]  activated output (bf16)
    """
    if not _USE_TRITON:
        return F.leaky_relu(F.linear(x, weight), negative_slope=0.5).square()

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])  # [M, K]
    M, K = x_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    # weight is [N, K] contiguous. We want A[M,K] @ B[K,N].
    # B = weight.T, so: stride_bk = weight.stride(1), stride_bn = weight.stride(0)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _fused_mlp_up_proj_kernel[grid](
        x_2d, weight, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),  # transposed: stride_bk=inner, stride_bn=outer
        out.stride(0), out.stride(1),
    )

    return out.reshape(*orig_shape[:-1], N)


# ─── Kernel 2: fused_mlp_down_proj ───────────────────────────────��───────────
#
# C[M,N] = residual[M,N] + mlp_scale[N] * ( A[M,K] @ B[K,N] )
#
# A = hidden (activated MLP intermediate),  B = down_w.T (via strides)
# K = mlp_dim (1536),  N = model_dim (512)
# M = B*T (batch tokens)
#
# Epilogue fuses: scale broadcast + residual add into the GEMM,
# eliminating the 512-dim GEMM output HBM write + separate scale+add kernel.
#
# Configs: K=1536 is larger → add BLOCK_K=128 options.
# N=512 is small → BLOCK_N=64 or 128 preferred. BLOCK_N=256 wastes tiles.

_DOWN_PROJ_CONFIGS = [
    # Core — same winners as Kernel 1 but with K-axis variants
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                  num_stages=5, num_warps=2),
    # Larger BLOCK_K for K=1536 (fewer loop iterations)
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                  num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                  num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                  num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_SIZE_M': 8},
                  num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_SIZE_M': 4},
                  num_stages=3, num_warps=4),
    # Small-tile fallback
    triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
                  num_stages=4, num_warps=4),
]


@triton.autotune(configs=_DOWN_PROJ_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_mlp_down_proj_kernel(
    # Pointers
    a_ptr,        # hidden      [M, K]  (contiguous, K=1536)
    b_ptr,        # weight      [N, K]  (row-major, transposed via strides)
    c_ptr,        # output      [M, N]  (contiguous, N=512)
    scale_ptr,    # mlp_scale   [N]     (per-dim scale vector)
    res_ptr,      # residual    [M, N]  (same layout as output)
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_rm, stride_rn,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """GEMM + scale + residual add, fused in one kernel.

    Computes C = residual + mlp_scale * (A @ B) where:
      A: [M, K] hidden activations (bf16, K=1536)
      B: [K, N] transposed weight (bf16, from [N, K] via strides)
      mlp_scale: [N] per-dimension scale (bf16)
      residual: [M, N] skip connection (bf16)
      C: [M, N] output (bf16)

    Accumulation in float32.
    """
    # ── Program ID -> tile coordinates (grouped for L2 locality) ──
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── Pointers for this tile ──
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # ── Main GEMM loop ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── Fused epilogue: out = residual + mlp_scale * gemm_out ──
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Load per-dim scale [BLOCK_N]
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    # Load residual tile [BLOCK_M, BLOCK_N]
    res_ptrs = res_ptr + offs_cm[:, None] * stride_rm + offs_cn[None, :] * stride_rn
    residual = tl.load(res_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    # Fused scale + add (in float32, then cast)
    out = residual + scale[None, :] * acc

    # ── Store output ──
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


def fused_mlp_down_proj(
    hidden: Tensor, weight: Tensor, mlp_scale: Tensor, residual: Tensor,
) -> Tensor:
    """Fused down-projection: out = residual + mlp_scale * (hidden @ weight.T)

    Args:
        hidden:    [*, K]  activated MLP intermediate (bf16, K=1536)
        weight:    [N, K]  down-projection weight (bf16)
        mlp_scale: [N]    per-dimension scale vector (bf16)
        residual:  [*, N]  skip connection input (bf16)

    Returns:
        [*, N]  output (bf16)
    """
    if not _USE_TRITON:
        return residual + mlp_scale * F.linear(hidden, weight)

    orig_shape = hidden.shape
    hidden_2d = hidden.reshape(-1, hidden.shape[-1])  # [M, K]
    res_2d = residual.reshape(-1, residual.shape[-1])  # [M, N]
    M, K = hidden_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), device=hidden.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    _fused_mlp_down_proj_kernel[grid](
        hidden_2d, weight, out, mlp_scale, res_2d,
        M, N, K,
        hidden_2d.stride(0), hidden_2d.stride(1),
        weight.stride(1), weight.stride(0),  # transposed
        out.stride(0), out.stride(1),
        res_2d.stride(0), res_2d.stride(1),
    )

    return out.reshape(*orig_shape[:-1], N)


# ─── Correctness + Benchmark Harness ────────────────────────────────────────

def _get_best_config(kernel, M, N, K):
    """Extract the winning autotune config for given dimensions."""
    key = (M, N, K)
    if hasattr(kernel, 'cache'):
        for cached_key, config in kernel.cache.items():
            if key in str(cached_key) or True:
                return config
    if hasattr(kernel, 'best_config'):
        return kernel.best_config
    return None


if __name__ == '__main__':
    import time
    torch.manual_seed(1337)
    device = 'cuda'

    K, N = 512, 1536
    N_ITER = 500
    batch_sizes = [4096, 8192, 16384]

    print("=" * 72)
    print(f"Kernel 1: fused_mlp_up_proj  [{torch.cuda.get_device_name()}]")
    print(f"GEMM: [M, {K}] @ [{K}, {N}]  +  LeakyReLU(0.5)^2 epilogue")
    print("=" * 72)

    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    def cublas_path(x, w):
        return F.leaky_relu(F.linear(x, w), negative_slope=0.5).square()

    results = []

    for M in batch_sizes:
        print(f"\n--- M = {M} ---")
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # ── Correctness ──
        ref = cublas_path(x, w)
        # Reset autotune cache so it re-tunes for this M
        _fused_mlp_up_proj_kernel.cache.clear()
        out = fused_mlp_up_proj(x, w)
        cos = F.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0)
        max_diff = (ref.float() - out.float()).abs().max().item()
        rel_err = ((ref.float() - out.float()).abs() / (ref.float().abs() + 1e-8)).max().item()
        print(f"  Correctness: cos_sim={cos.item():.6f}  max_abs_diff={max_diff:.2f}  max_rel_err={rel_err:.6f}")

        # ── Extract winning config ──
        best = None
        try:
            for k, v in _fused_mlp_up_proj_kernel.cache.items():
                best = v
                break
        except Exception:
            pass
        if best is not None:
            kw = best.kwargs
            print(f"  Autotune winner: BLOCK_M={kw['BLOCK_M']} BLOCK_N={kw['BLOCK_N']} "
                  f"BLOCK_K={kw['BLOCK_K']} GROUP_SIZE_M={kw['GROUP_SIZE_M']} "
                  f"stages={best.num_stages} warps={best.num_warps}")
        else:
            print("  Autotune winner: (could not extract)")

        # ── Benchmark ──
        # Warmup
        for _ in range(30):
            _ = cublas_path(x, w)
            _ = fused_mlp_up_proj(x, w)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = cublas_path(x, w)
        torch.cuda.synchronize()
        cublas_ms = (time.perf_counter() - t0) / N_ITER * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = fused_mlp_up_proj(x, w)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) / N_ITER * 1000

        speedup = cublas_ms / triton_ms
        savings = cublas_ms - triton_ms
        print(f"  cuBLAS + separate act:  {cublas_ms:.4f} ms")
        print(f"  Triton fused:           {triton_ms:.4f} ms")
        print(f"  Speedup:                {speedup:.2f}x")
        print(f"  Savings per call:       {savings:.4f} ms")

        results.append((M, cublas_ms, triton_ms, speedup, savings, cos.item()))

    # ── Summary ──
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n{'M':>6} | {'cuBLAS+act':>11} | {'Triton fused':>12} | {'Speedup':>8} | {'Save/call':>9} | {'cos_sim':>8}")
    print("-" * 72)
    for M, cb, tr, sp, sv, cs in results:
        print(f"{M:>6} | {cb:>9.4f}ms | {tr:>10.4f}ms | {sp:>7.2f}x | {sv:>7.4f}ms | {cs:>8.6f}")

    print(f"\nProjected per-step savings on H100 (22 MLP up-proj calls):")
    parts = []
    for M, _, _, _, sv, _ in results:
        parts.append(f"M={M}: {sv * 22:.2f} ms")
    print("  " + " | ".join(parts))

    # ── Go/no-go gate ──
    all_fast = all(sp >= 1.2 for _, _, _, sp, _, _ in results)
    all_correct = all(cs > 0.9999 for _, _, _, _, _, cs in results)
    print(f"\nGo/no-go: speedup >= 1.2x at all M? {all_fast}  |  cos_sim > 0.9999 at all M? {all_correct}")
    if all_fast and all_correct:
        print(">>> GATE PASSED")

    # ====================================================================
    # KERNEL 2: fused_mlp_down_proj
    # ====================================================================
    K2, N2 = 1536, 512  # down projection: [M, 1536] @ [1536, 512]

    print("\n\n" + "=" * 72)
    print(f"Kernel 2: fused_mlp_down_proj  [{torch.cuda.get_device_name()}]")
    print(f"GEMM: [M, {K2}] @ [{K2}, {N2}]  +  scale*out + residual epilogue")
    print("=" * 72)

    w_down = torch.randn(N2, K2, device=device, dtype=torch.bfloat16)
    mlp_scale = torch.randn(N2, device=device, dtype=torch.bfloat16)

    def cublas_down_path(hidden, w_down, mlp_scale, residual):
        return residual + mlp_scale[None, :] * F.linear(hidden, w_down)

    results2 = []

    for M in batch_sizes:
        print(f"\n--- M = {M} ---")
        hidden = torch.randn(M, K2, device=device, dtype=torch.bfloat16)
        residual = torch.randn(M, N2, device=device, dtype=torch.bfloat16)

        # ── Correctness ──
        ref2 = cublas_down_path(hidden, w_down, mlp_scale, residual)
        _fused_mlp_down_proj_kernel.cache.clear()
        out2 = fused_mlp_down_proj(hidden, w_down, mlp_scale, residual)
        cos2 = F.cosine_similarity(ref2.flatten().float(), out2.flatten().float(), dim=0)
        max_diff2 = (ref2.float() - out2.float()).abs().max().item()
        rel_err2 = ((ref2.float() - out2.float()).abs() / (ref2.float().abs() + 1e-8)).max().item()
        print(f"  Correctness: cos_sim={cos2.item():.6f}  max_abs_diff={max_diff2:.2f}  max_rel_err={rel_err2:.6f}")

        # ── Extract winning config ──
        best2 = None
        try:
            for k, v in _fused_mlp_down_proj_kernel.cache.items():
                best2 = v
                break
        except Exception:
            pass
        if best2 is not None:
            kw = best2.kwargs
            print(f"  Autotune winner: BLOCK_M={kw['BLOCK_M']} BLOCK_N={kw['BLOCK_N']} "
                  f"BLOCK_K={kw['BLOCK_K']} GROUP_SIZE_M={kw['GROUP_SIZE_M']} "
                  f"stages={best2.num_stages} warps={best2.num_warps}")

        # ── Benchmark ──
        for _ in range(30):
            _ = cublas_down_path(hidden, w_down, mlp_scale, residual)
            _ = fused_mlp_down_proj(hidden, w_down, mlp_scale, residual)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = cublas_down_path(hidden, w_down, mlp_scale, residual)
        torch.cuda.synchronize()
        cb_ms = (time.perf_counter() - t0) / N_ITER * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = fused_mlp_down_proj(hidden, w_down, mlp_scale, residual)
        torch.cuda.synchronize()
        tr_ms = (time.perf_counter() - t0) / N_ITER * 1000

        speedup2 = cb_ms / tr_ms
        savings2 = cb_ms - tr_ms
        print(f"  cuBLAS + separate scale+res: {cb_ms:.4f} ms")
        print(f"  Triton fused:                {tr_ms:.4f} ms")
        print(f"  Speedup:                     {speedup2:.2f}x")
        print(f"  Savings per call:            {savings2:.4f} ms")

        results2.append((M, cb_ms, tr_ms, speedup2, savings2, cos2.item()))

    # ── Kernel 2 Summary ──
    print("\n" + "=" * 72)
    print("KERNEL 2 SUMMARY")
    print("=" * 72)
    print(f"\n{'M':>6} | {'cuBLAS+s+r':>11} | {'Triton fused':>12} | {'Speedup':>8} | {'Save/call':>9} | {'cos_sim':>8}")
    print("-" * 72)
    for M, cb, tr, sp, sv, cs in results2:
        print(f"{M:>6} | {cb:>9.4f}ms | {tr:>10.4f}ms | {sp:>7.2f}x | {sv:>7.4f}ms | {cs:>8.6f}")

    print(f"\nProjected per-step savings on H100 (22 MLP down-proj calls):")
    parts2 = []
    for M, _, _, _, sv, _ in results2:
        parts2.append(f"M={M}: {sv * 22:.2f} ms")
    print("  " + " | ".join(parts2))

    # ── Combined MLP savings ──
    print("\n" + "=" * 72)
    print("COMBINED MLP SAVINGS (Kernel 1 + Kernel 2, 22 calls each)")
    print("=" * 72)
    for i, M in enumerate(batch_sizes):
        s1 = results[i][4]   # Kernel 1 savings per call
        s2 = results2[i][4]  # Kernel 2 savings per call
        total = (s1 + s2) * 22
        pct = total / 86.7 * 100 if total > 0 else 0  # vs 86.7ms SOTA step
        print(f"  M={M}: {total:.2f} ms/step  ({pct:.1f}% of 86.7ms SOTA step)")
