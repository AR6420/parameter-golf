"""
CUTLASS/Triton Feasibility Spike for GEMM Boundary Fusion

Tests multiple approaches for fusing element-wise ops into GEMM boundaries:
1. Inductor CUTLASS EVT (internal, enable config flag)
2. Triton GEMM with inline epilogue (autotuned)
3. Baseline: cuBLAS + separate Triton element-wise kernel (current SOTA)

Target: out = LeakyReLU(x @ W, 0.5).square()  for [4096, 512] @ [512, 1536]
"""
import os, time, math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.manual_seed(1337)

# Shapes matching SOTA MLP
M = 4096      # batch tokens
K = 512       # model_dim
N = 1536      # mlp_dim (3x)

device = 'cuda'
dtype = torch.bfloat16

x = torch.randn(M, K, device=device, dtype=dtype)
w = torch.randn(N, K, device=device, dtype=dtype)  # weight [N, K] for F.linear
w_t = w.T.contiguous()  # [K, N] for mm

# ============================================================
# APPROACH 1: Inductor CUTLASS EVT (internal torch.compile path)
# ============================================================
# Enable CUTLASS epilogue fusion in Inductor
torch._inductor.config.cutlass.cutlass_epilogue_fusion_enabled = True
torch._inductor.config.epilogue_fusion = True

# Also try forcing CUTLASS backend
print("=" * 70)
print("APPROACH 1: torch.compile with CUTLASS epilogue fusion enabled")
print("=" * 70)

def mlp_up_fused(x, w):
    """MLP up projection with fused LeakyReLU^2 epilogue"""
    return F.leaky_relu(F.linear(x, w), negative_slope=0.5).square()

def mlp_up_fused_bias(x, w, bias):
    """With bias to trigger addmm path"""
    return F.leaky_relu(F.linear(x, w, bias), negative_slope=0.5).square()

# Reference output
ref = mlp_up_fused(x, w)

# Test default compile
torch._dynamo.reset()
compiled_default = torch.compile(mlp_up_fused, dynamic=False)
with torch.no_grad():
    out1 = compiled_default(x, w)
cos1 = F.cosine_similarity(ref.flatten().float(), out1.flatten().float(), dim=0)
print(f"  Default compile cos_sim: {cos1.item():.6f}")

# Test max-autotune compile
torch._dynamo.reset()
compiled_autotune = torch.compile(mlp_up_fused, mode='max-autotune', dynamic=False)
with torch.no_grad():
    out2 = compiled_autotune(x, w)
cos2 = F.cosine_similarity(ref.flatten().float(), out2.flatten().float(), dim=0)
print(f"  max-autotune compile cos_sim: {cos2.item():.6f}")

# ============================================================
# APPROACH 2: Triton GEMM with inline epilogue
# ============================================================
print()
print("=" * 70)
print("APPROACH 2: Triton GEMM with fused LeakyReLU^2 epilogue")
print("=" * 70)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_leaky_relu_sq_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """GEMM + LeakyReLU(0.5) + square fused in one kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # FUSED EPILOGUE: LeakyReLU(0.5) + square
    # LeakyReLU: out = x if x > 0 else 0.5 * x
    acc = tl.where(acc > 0, acc, acc * 0.5)
    acc = acc * acc  # square

    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm_leaky_relu_sq(a, b_t):
    """a: [M, K], b_t: [K, N] -> out: [M, N] = LeakyReLU(a @ b_t, 0.5)^2"""
    M, K = a.shape
    K2, N = b_t.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_leaky_relu_sq_kernel[grid](
        a, b_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# Correctness
out_triton = triton_gemm_leaky_relu_sq(x, w_t)
cos_triton = F.cosine_similarity(ref.flatten().float(), out_triton.flatten().float(), dim=0)
max_diff = (ref.float() - out_triton.float()).abs().max().item()
print(f"  Triton GEMM+epilogue cos_sim: {cos_triton.item():.6f}")
print(f"  Max abs diff: {max_diff:.6f}")

# ============================================================
# APPROACH 3: Baseline (cuBLAS mm + separate activation)
# ============================================================
print()
print("=" * 70)
print("BENCHMARKS")
print("=" * 70)

def cublas_then_act(x, w_t):
    out = torch.mm(x, w_t)
    return F.leaky_relu(out, negative_slope=0.5).square()

# Warmup all paths
for _ in range(20):
    _ = cublas_then_act(x, w_t)
    _ = triton_gemm_leaky_relu_sq(x, w_t)
    with torch.no_grad():
        _ = compiled_default(x, w)
        _ = compiled_autotune(x, w)
torch.cuda.synchronize()

N_ITER = 500

# Benchmark cuBLAS only (no activation)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = torch.mm(x, w_t)
torch.cuda.synchronize()
cublas_only_ms = (time.perf_counter() - t0) / N_ITER * 1000

# Benchmark cuBLAS + separate activation (CURRENT SOTA)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = cublas_then_act(x, w_t)
torch.cuda.synchronize()
cublas_act_ms = (time.perf_counter() - t0) / N_ITER * 1000

# Benchmark Triton GEMM + fused epilogue
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = triton_gemm_leaky_relu_sq(x, w_t)
torch.cuda.synchronize()
triton_fused_ms = (time.perf_counter() - t0) / N_ITER * 1000

# Benchmark torch.compile (default)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    with torch.no_grad():
        _ = compiled_default(x, w)
torch.cuda.synchronize()
compile_default_ms = (time.perf_counter() - t0) / N_ITER * 1000

# Benchmark torch.compile (max-autotune)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    with torch.no_grad():
        _ = compiled_autotune(x, w)
torch.cuda.synchronize()
compile_autotune_ms = (time.perf_counter() - t0) / N_ITER * 1000

print(f"\n  Shape: [{M}, {K}] @ [{K}, {N}] -> [{M}, {N}]")
print(f"  Dtype: {dtype}")
print(f"  Device: {torch.cuda.get_device_name()}")
print()
print(f"  cuBLAS mm only:                 {cublas_only_ms:.4f} ms")
print(f"  cuBLAS + separate activation:   {cublas_act_ms:.4f} ms  (SOTA baseline)")
print(f"  Triton GEMM + fused epilogue:   {triton_fused_ms:.4f} ms  ({triton_fused_ms/cublas_act_ms:.2f}x)")
print(f"  torch.compile (default):        {compile_default_ms:.4f} ms  ({compile_default_ms/cublas_act_ms:.2f}x)")
print(f"  torch.compile (max-autotune):   {compile_autotune_ms:.4f} ms  ({compile_autotune_ms/cublas_act_ms:.2f}x)")
print()
activation_overhead = cublas_act_ms - cublas_only_ms
print(f"  Activation overhead (HBM roundtrip): {activation_overhead:.4f} ms")
print(f"  Triton GEMM overhead vs cuBLAS:      {triton_fused_ms - cublas_only_ms:.4f} ms")
print(f"  Net savings from fusion:             {cublas_act_ms - triton_fused_ms:.4f} ms")

# ============================================================
# Also test the DOWN projection: [4096, 1536] @ [1536, 512]
# ============================================================
print()
print("=" * 70)
print("DOWN PROJECTION: [{}, {}] @ [{}, {}]".format(M, N, N, K))
print("=" * 70)

x_big = torch.randn(M, N, device=device, dtype=dtype)
w_down = torch.randn(K, N, device=device, dtype=dtype)
w_down_t = w_down.T.contiguous()

# For down projection the epilogue is: scale * out + residual
# Different epilogue, but let's test GEMM speed first

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = torch.mm(x_big, w_down_t)
torch.cuda.synchronize()
cublas_down_ms = (time.perf_counter() - t0) / N_ITER * 1000
print(f"  cuBLAS mm [{M},{N}]@[{N},{K}]:   {cublas_down_ms:.4f} ms")

# Triton down projection (just GEMM, no epilogue)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_residual_kernel(
    a_ptr, b_ptr, c_ptr, scale_ptr, res_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """GEMM + scale*out + residual fused in one kernel."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # FUSED EPILOGUE: scale * out + residual
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Load scale [N] and residual [M, N]
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    res_ptrs = res_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    residual = tl.load(res_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    out = residual + scale[None, :] * acc
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


def triton_gemm_scale_residual(a, b_t, scale, residual):
    M_loc, K_loc = a.shape
    K2, N_loc = b_t.shape
    assert K_loc == K2
    c = torch.empty_like(residual)
    grid = lambda META: (triton.cdiv(M_loc, META['BLOCK_M']) * triton.cdiv(N_loc, META['BLOCK_N']),)
    gemm_scale_residual_kernel[grid](
        a, b_t, c, scale, residual,
        M_loc, N_loc, K_loc,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# Test down projection with fused scale + residual
scale_down = torch.randn(K, device=device, dtype=dtype)
residual = torch.randn(M, K, device=device, dtype=dtype)

# Reference
ref_down = residual + scale_down[None, :] * torch.mm(x_big, w_down_t)
out_down = triton_gemm_scale_residual(x_big, w_down_t, scale_down, residual)
cos_down = F.cosine_similarity(ref_down.flatten().float(), out_down.flatten().float(), dim=0)
print(f"  Triton GEMM+scale+residual cos_sim: {cos_down.item():.6f}")

# Benchmark
def cublas_scale_residual(a, b_t, scale, res):
    return res + scale[None, :] * torch.mm(a, b_t)

for _ in range(20):
    _ = cublas_scale_residual(x_big, w_down_t, scale_down, residual)
    _ = triton_gemm_scale_residual(x_big, w_down_t, scale_down, residual)
torch.cuda.synchronize()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = cublas_scale_residual(x_big, w_down_t, scale_down, residual)
torch.cuda.synchronize()
cublas_sr_ms = (time.perf_counter() - t0) / N_ITER * 1000

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = triton_gemm_scale_residual(x_big, w_down_t, scale_down, residual)
torch.cuda.synchronize()
triton_sr_ms = (time.perf_counter() - t0) / N_ITER * 1000

print(f"  cuBLAS + separate scale+residual: {cublas_sr_ms:.4f} ms")
print(f"  Triton GEMM + fused scale+res:    {triton_sr_ms:.4f} ms  ({triton_sr_ms/cublas_sr_ms:.2f}x)")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
