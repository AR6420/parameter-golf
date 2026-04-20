"""
ForgeFuse Phase 2B — W8A8 Triton kernels + autograd wrappers.

Per-channel symmetric INT8 weight quantization + per-token dynamic INT8
activation quantization. Uses Hopper/Blackwell INT8 tensor cores via
`tl.dot(int8, int8, out_dtype=tl.int32)` — unlocks the 2x TFLOPs ratio
over BF16 on H100 (1979 vs 989 TFLOPs theoretical).

STE backward: both weight- and activation-quantization are treated as
identity on backward. bf16 master weight + original bf16 activation are
used for gradient computation. Master weights stay learnable in banks.

Epilogue precision rule (locked in by user):
  acc_fp32 = acc_int32.to(tl.float32)
  acc_fp32 = acc_fp32 * x_scale[:, None].to(tl.float32) * w_scale[None, :].to(tl.float32)
  # then user epilogue (LeakyReLU^2, scale+residual) in fp32
  out = acc_fp32.to(tl.bfloat16)

Phase 2B-MVP scope: MLP_up kernel + wrapper only. Other 3 variants
will be cloned after MLP_up passes correctness and timing review.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


# ─── Quantization utilities ──────────────────────────────────────────────────

def quantize_w8_per_channel(w_bf16: Tensor) -> tuple[Tensor, Tensor]:
    """Per-output-channel symmetric INT8 quant — identical math to W8A16's quantizer.

    w_bf16: [N, K] bf16
    returns (w_int8 [N, K] int8, w_scale [N] bf16)
    """
    amax = w_bf16.abs().amax(dim=-1, keepdim=True)             # [N, 1]
    scale_fp = (amax / 127.0).clamp(min=1e-8).float()
    w_int8 = (w_bf16.float() / scale_fp).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale_fp.squeeze(-1).to(torch.bfloat16)


def quantize_a8_per_token(x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-token symmetric INT8 quant — dynamic, one scale per row of the [M, K] view.

    x:         [..., K] bf16 (any leading dims collapse into M at the call site)
    returns   (x_int8 [M, K] int8, x_scale [M] bf16)
    """
    assert x.dim() >= 2, "quantize_a8_per_token expects rank-2+ input"
    x_flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    amax = x_flat.abs().amax(dim=-1, keepdim=True)             # [M, 1]
    scale_fp = (amax / 127.0).clamp(min=1e-8).float()
    x_int8 = (x_flat.float() / scale_fp).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale_fp.squeeze(-1).to(torch.bfloat16)


# ─── Kernel: MLP_up — W8A8 + LeakyReLU(0.5)^2 epilogue ───────────────────────

@triton.jit
def _fused_mlp_up_w8a8_kernel(
    a_ptr,        # int8 [M, K]   pre-quantized activations
    b_ptr,        # int8 [N, K]   pre-quantized weights (loaded as [K, N] via strides)
    xs_ptr,       # bf16 [M]      per-token activation scale
    ws_ptr,       # bf16 [N]      per-channel weight scale
    c_ptr,        # bf16 [M, N]   output
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """out[m, n] = LeakyReLU( x_scale[m] * w_scale[n] * sum_k( x_int8[m,k] * w_int8[n,k] ), 0.5 ) ** 2

    INT8 tensor-core GEMM via tl.dot(int8, int8, out_dtype=tl.int32).
    """
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

    # --- INT8 tensor-core GEMM, int32 accumulator ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)                                    # int8 [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs)                                    # int8 [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b, out_dtype=tl.int32)                # int32 GEMM
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Epilogue in fp32 (int32 -> fp32 FIRST, scales applied separately) ---
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)   # [BLOCK_M]
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)   # [BLOCK_N]
    out = acc.to(tl.float32)
    out = out * xs[:, None]
    out = out * ws[None, :]
    # LeakyReLU(0.5)^2 — same as Sub #2 / W8A16
    out = tl.where(out > 0, out, 0.5 * out)
    out = out * out

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


# ─── Launcher ────────────────────────────────────────────────────────────────

def _launch_mlp_up_w8a8(x_int8: Tensor, w_int8: Tensor,
                        x_scale: Tensor, w_scale: Tensor) -> Tensor:
    assert x_int8.dtype == torch.int8 and w_int8.dtype == torch.int8
    M, K = x_int8.shape
    N, K2 = w_int8.shape
    assert K == K2, f"K mismatch: x {x_int8.shape} vs w {w_int8.shape}"
    out = torch.empty((M, N), device=x_int8.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_mlp_up_w8a8_kernel[grid](
        x_int8, w_int8, x_scale, w_scale, out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),        # stride trick: view [N,K] as [K,N]
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ─── Autograd wrapper with STE backward ──────────────────────────────────────

class _MLPUp_W8A8(torch.autograd.Function):
    """out = LeakyReLU(x @ dequant(w).T, 0.5) ** 2, with per-token A + per-channel W int8.
    STE backward: quant is identity on both x and w — gradients use bf16 master tensors."""
    @staticmethod
    def forward(ctx, x, weight_bf16):
        orig_shape = x.shape
        # Per-token activation quant
        x_int8, x_scale = quantize_a8_per_token(x)             # [M, K] int8, [M] bf16
        # Per-channel weight quant (same math as W8A16)
        w_int8, w_scale = quantize_w8_per_channel(weight_bf16.to(torch.bfloat16))
        # Fused int8 GEMM + LeakyReLU^2 epilogue
        out_2d = _launch_mlp_up_w8a8(x_int8, w_int8, x_scale, w_scale)
        ctx.save_for_backward(x, weight_bf16)
        return out_2d.reshape(*orig_shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        # STE: backward treats both quant steps as identity.
        # Math matches the W8A16 MLP_up backward exactly (same epilogue).
        x, weight = ctx.saved_tensors
        w_cast = weight.to(grad_out.dtype)
        x_cast = x.to(grad_out.dtype)
        pre = F.linear(x_cast, w_cast)
        activated = F.leaky_relu(pre, negative_slope=0.5)
        leaky_grad = torch.where(pre > 0, torch.ones_like(pre), torch.full_like(pre, 0.5))
        grad_pre = grad_out * 2.0 * activated * leaky_grad
        grad_x = F.linear(grad_pre, w_cast.T.contiguous()) if x.requires_grad else None
        gp_2d = grad_pre.reshape(-1, grad_pre.shape[-1])
        x_2d = x_cast.reshape(-1, x_cast.shape[-1])
        grad_w = gp_2d.T @ x_2d if weight.requires_grad else None
        return grad_x, grad_w


def fused_mlp_up_w8a8(x: Tensor, weight_bf16: Tensor) -> Tensor:
    """out = LeakyReLU(x @ dequant(w).T, 0.5) ** 2 via per-token A8 + per-channel W8 INT8 GEMM."""
    return _MLPUp_W8A8.apply(x, weight_bf16)


# ─── Kernel: QKV — plain dequant, no fused op beyond int32→bf16 ──────────────

@triton.jit
def _fused_qkv_w8a8_kernel(
    a_ptr, b_ptr, xs_ptr, ws_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """out[m, n] = x_scale[m] * w_scale[n] * sum_k( x_int8[m,k] * w_int8[n,k] )"""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    out = acc.to(tl.float32)
    out = out * xs[:, None]
    out = out * ws[None, :]

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


# ─── Kernel: OutProj — dequant + attn_scale * residual ───────────────────────

@triton.jit
def _fused_out_proj_w8a8_kernel(
    a_ptr, b_ptr, xs_ptr, ws_ptr, attn_scale_ptr, res_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_rm, stride_rn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """out = residual + attn_scale[None, :] * (x_scale[m] * w_scale[n] * int8_gemm)"""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    attn_scale = tl.load(attn_scale_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    res_ptrs = res_ptr + offs_cm[:, None] * stride_rm + offs_cn[None, :] * stride_rn
    residual = tl.load(res_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    out = acc.to(tl.float32)
    out = out * xs[:, None]
    out = out * ws[None, :]
    out = residual + attn_scale[None, :] * out

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


# ─── Kernel: MLP_down — dequant + mlp_scale * residual ───────────────────────

@triton.jit
def _fused_mlp_down_w8a8_kernel(
    a_ptr, b_ptr, xs_ptr, ws_ptr, mlp_scale_ptr, res_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_rm, stride_rn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """out = residual + mlp_scale[None, :] * (x_scale[m] * w_scale[n] * int8_gemm)"""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    mlp_scale = tl.load(mlp_scale_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    res_ptrs = res_ptr + offs_cm[:, None] * stride_rm + offs_cn[None, :] * stride_rn
    residual = tl.load(res_ptrs, mask=c_mask, other=0.0).to(tl.float32)

    out = acc.to(tl.float32)
    out = out * xs[:, None]
    out = out * ws[None, :]
    out = residual + mlp_scale[None, :] * out

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=c_mask)


# ─── Launchers ───────────────────────────────────────────────────────────────

def _launch_qkv_w8a8(x_int8: Tensor, w_int8: Tensor,
                     x_scale: Tensor, w_scale: Tensor) -> Tensor:
    M, K = x_int8.shape
    N = w_int8.shape[0]
    out = torch.empty((M, N), device=x_int8.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_qkv_w8a8_kernel[grid](
        x_int8, w_int8, x_scale, w_scale, out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _launch_out_proj_w8a8(x_int8: Tensor, w_int8: Tensor,
                          x_scale: Tensor, w_scale: Tensor,
                          attn_scale: Tensor, residual: Tensor) -> Tensor:
    M, K = x_int8.shape
    N = w_int8.shape[0]
    out = torch.empty((M, N), device=x_int8.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_out_proj_w8a8_kernel[grid](
        x_int8, w_int8, x_scale, w_scale, attn_scale, residual, out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _launch_mlp_down_w8a8(x_int8: Tensor, w_int8: Tensor,
                          x_scale: Tensor, w_scale: Tensor,
                          mlp_scale: Tensor, residual: Tensor) -> Tensor:
    M, K = x_int8.shape
    N = w_int8.shape[0]
    out = torch.empty((M, N), device=x_int8.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_mlp_down_w8a8_kernel[grid](
        x_int8, w_int8, x_scale, w_scale, mlp_scale, residual, out,
        M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(1), w_int8.stride(0),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ─── Autograd wrappers ───────────────────────────────────────────────────────

class _QKV_W8A8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_bf16):
        orig_shape = x.shape
        x_int8, x_scale = quantize_a8_per_token(x)
        w_int8, w_scale = quantize_w8_per_channel(weight_bf16.to(torch.bfloat16))
        out_2d = _launch_qkv_w8a8(x_int8, w_int8, x_scale, w_scale)
        ctx.save_for_backward(x, weight_bf16)
        return out_2d.reshape(*orig_shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        w_cast = weight.to(grad_out.dtype)
        x_cast = x.to(grad_out.dtype)
        grad_x = F.linear(grad_out, w_cast.T.contiguous()) if x.requires_grad else None
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])
        x_2d = x_cast.reshape(-1, x_cast.shape[-1])
        grad_w = go_2d.T @ x_2d if weight.requires_grad else None
        return grad_x, grad_w


class _OutProj_W8A8(torch.autograd.Function):
    """out = residual + attn_scale * (x @ dequant(w).T)"""
    @staticmethod
    def forward(ctx, x, weight_bf16, attn_scale, residual):
        orig_shape = x.shape
        x_int8, x_scale = quantize_a8_per_token(x)
        w_int8, w_scale = quantize_w8_per_channel(weight_bf16.to(torch.bfloat16))
        res_2d = residual.reshape(-1, residual.shape[-1]).to(torch.bfloat16).contiguous()
        out_2d = _launch_out_proj_w8a8(x_int8, w_int8, x_scale, w_scale,
                                       attn_scale.to(torch.bfloat16), res_2d)
        ctx.save_for_backward(x, weight_bf16, attn_scale)
        return out_2d.reshape(*orig_shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, attn_scale = ctx.saved_tensors
        grad_residual = grad_out
        w_cast = weight.to(grad_out.dtype)
        x_cast = x.to(grad_out.dtype)
        scaled = attn_scale.to(grad_out.dtype) * grad_out
        grad_x = F.linear(scaled, w_cast.T.contiguous()) if x.requires_grad else None
        sg_2d = scaled.reshape(-1, scaled.shape[-1])
        x_2d = x_cast.reshape(-1, x_cast.shape[-1])
        grad_w = sg_2d.T @ x_2d if weight.requires_grad else None
        gemm_out = F.linear(x_cast, w_cast)
        grad_attn_scale = (grad_out * gemm_out).reshape(-1, grad_out.shape[-1]).sum(dim=0) \
            if attn_scale.requires_grad else None
        return grad_x, grad_w, grad_attn_scale, grad_residual


class _MLPDown_W8A8(torch.autograd.Function):
    """out = residual + mlp_scale * (hidden @ dequant(w).T)"""
    @staticmethod
    def forward(ctx, hidden, weight_bf16, mlp_scale, residual):
        orig_shape = hidden.shape
        h_int8, h_scale = quantize_a8_per_token(hidden)
        w_int8, w_scale = quantize_w8_per_channel(weight_bf16.to(torch.bfloat16))
        res_2d = residual.reshape(-1, residual.shape[-1]).to(torch.bfloat16).contiguous()
        out_2d = _launch_mlp_down_w8a8(h_int8, w_int8, h_scale, w_scale,
                                       mlp_scale.to(torch.bfloat16), res_2d)
        ctx.save_for_backward(hidden, weight_bf16, mlp_scale)
        return out_2d.reshape(*orig_shape[:-1], weight_bf16.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, mlp_scale = ctx.saved_tensors
        grad_residual = grad_out
        w_cast = weight.to(grad_out.dtype)
        h_cast = hidden.to(grad_out.dtype)
        scaled = mlp_scale.to(grad_out.dtype) * grad_out
        grad_hidden = F.linear(scaled, w_cast.T.contiguous()) if hidden.requires_grad else None
        sg_2d = scaled.reshape(-1, scaled.shape[-1])
        h_2d = h_cast.reshape(-1, h_cast.shape[-1])
        grad_w = sg_2d.T @ h_2d if weight.requires_grad else None
        gemm_out = F.linear(h_cast, w_cast)
        grad_mlp_scale = (grad_out * gemm_out).reshape(-1, grad_out.shape[-1]).sum(dim=0) \
            if mlp_scale.requires_grad else None
        return grad_hidden, grad_w, grad_mlp_scale, grad_residual


# ─── Public entry points ─────────────────────────────────────────────────────

def fused_qkv_w8a8(x: Tensor, weight_bf16: Tensor) -> Tensor:
    """out = x @ dequant(weight).T — per-token A8 + per-channel W8."""
    return _QKV_W8A8.apply(x, weight_bf16)


def fused_out_proj_w8a8(x: Tensor, weight_bf16: Tensor,
                        attn_scale: Tensor, residual: Tensor) -> Tensor:
    """out = residual + attn_scale * (x @ dequant(weight).T)"""
    return _OutProj_W8A8.apply(x, weight_bf16, attn_scale, residual)


def fused_mlp_down_w8a8(hidden: Tensor, weight_bf16: Tensor,
                        mlp_scale: Tensor, residual: Tensor) -> Tensor:
    """out = residual + mlp_scale * (hidden @ dequant(weight).T)"""
    return _MLPDown_W8A8.apply(hidden, weight_bf16, mlp_scale, residual)
