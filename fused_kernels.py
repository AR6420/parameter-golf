"""Compile-safe Triton kernels for pr-1493 MLP fusion.

This module is the canonical template for future kernels. Any new kernel
MUST follow the same pattern:
  1. @triton.jit compute kernel with @triton.autotune
  2. @triton_op wrapper with explicit dtype/contiguity asserts
  3. @register_fake for shape/dtype meta kernel
  4. Explicit setup_context + backward via torch.library.register_autograd
     (NOT torch.autograd.Function)

See docs/research/compile_compatibility_rules.md for the rules this
template enforces.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Autotune configs: target shape on pr-1493 defaults is
#   M = B * T = 8192, K = 512, N = 1024   (up-projection of MLP at mlp_mult=2)
# Tile menu covers small (for latency-sensitive local runs) through large
# (for H100 SM saturation). Hardcoded BLOCK_N=1536 was Sub #1's failure mode;
# autotune sidesteps that by benchmarking per-shape.
# ----------------------------------------------------------------------
_MLP_UP_CONFIGS = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_MLP_UP_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _fused_mlp_up_kernel(
    X_ptr, W_ptr, Y_ptr, YSQ_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_ysm, stride_ysn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """GEMM + relu + square.

    Inputs:  X [M, K] bf16,  W [N, K] fp32
    Outputs: Y [M, N] bf16 (post-relu),  YSQ [M, N] bf16 (Y * Y)
    Weight is cast fp32 -> bf16 inside the kernel to match CastedLinear's
    cast-at-matmul discipline without materializing a bf16 copy in HBM.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        w_bf16 = w.to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w_bf16), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    y_fp32 = tl.where(acc > 0.0, acc, 0.0)
    y_sq_fp32 = y_fp32 * y_fp32
    y_bf16 = y_fp32.to(tl.bfloat16)
    y_sq_bf16 = y_sq_fp32.to(tl.bfloat16)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_out = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    ysq_out = YSQ_ptr + offs_m[:, None] * stride_ysm + offs_n[None, :] * stride_ysn
    tl.store(y_out, y_bf16, mask=out_mask)
    tl.store(ysq_out, y_sq_bf16, mask=out_mask)


@triton_op("forgefuse::fused_mlp_up_relu2", mutates_args=())
def _fused_mlp_up_relu2_impl(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Internal triton_op. Returns (y_sq, y); use fused_mlp_up() as public API.

    Caller saves `y` for analytical backward; `y_sq` is the actual downstream
    input to the next projection.
    """
    assert x.is_cuda and w.is_cuda, "inputs must be CUDA"
    assert x.dtype == torch.bfloat16, f"activations must be bf16, got {x.dtype}"
    assert w.dtype == torch.float32, f"weight must be fp32 (CastedLinear native), got {w.dtype}"
    assert w.dim() == 2, f"weight must be 2D, got shape {tuple(w.shape)}"

    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x_flat.shape
    N, Kw = w.shape
    assert K == Kw, f"K mismatch: x has {K}, w has {Kw}"

    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    y_sq = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    wrap_triton(_fused_mlp_up_kernel)[grid](
        x_flat, w, y, y_sq,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        y_sq.stride(0), y_sq.stride(1),
    )
    out_shape = list(orig_shape[:-1]) + [N]
    return y_sq.reshape(out_shape), y.reshape(out_shape)


@_fused_mlp_up_relu2_impl.register_fake
def _fused_mlp_up_relu2_fake(x, w):
    N = w.shape[0]
    out_shape = list(x.shape[:-1]) + [N]
    return (
        torch.empty(out_shape, device=x.device, dtype=torch.bfloat16),
        torch.empty(out_shape, device=x.device, dtype=torch.bfloat16),
    )


def _setup_context(ctx, inputs, output):
    x, w = inputs
    _y_sq, y = output
    ctx.save_for_backward(x, w, y)


def _backward(ctx, grad_y_sq, grad_y):
    """Analytical backward — no requires_grad_(), no re-forward tricks.

    Forward:
        pre = x @ w.T                (bf16 activations * fp32 weight cast to bf16)
        y   = relu(pre)
        y_sq = y * y

    Saved: (x, w, y). We don't save `pre` — `y = relu(pre)` has all the info
    we need because d/dpre relu(pre) = 1[pre > 0] = 1[y > 0] (y is zero iff pre <= 0).

    Chain rule:
        d y_sq / d y          = 2y
        d y    / d pre        = 1[y > 0]
        grad_pre = grad_y_sq * 2y    (+ grad_y * 1[y>0]   if y is consumed)

    Since the public API discards `y`, grad_y is typically None; the `if`
    branch below handles the defensive case where autograd passes a zero tensor.
    """
    x, w, y = ctx.saved_tensors

    grad_pre = 2.0 * y * grad_y_sq
    if grad_y is not None:
        grad_pre = grad_pre + grad_y * (y > 0).to(grad_y.dtype)

    # Match CastedLinear semantics: cast fp32 weight to activation dtype at matmul.
    w_bf16 = w.to(grad_pre.dtype)

    orig_x_shape = x.shape
    N = grad_pre.shape[-1]
    K = w.shape[-1]
    grad_pre_flat = grad_pre.reshape(-1, N)
    x_flat = x.reshape(-1, K)

    # grad_x = grad_pre @ w_bf16   (bf16 matmul to match reference autograd)
    grad_x_flat = grad_pre_flat @ w_bf16
    grad_x = grad_x_flat.reshape(orig_x_shape)

    # grad_w = grad_pre^T @ x   (bf16 matmul, cast result to fp32 for CastedLinear.weight.grad)
    grad_w_bf16 = grad_pre_flat.t() @ x_flat
    grad_w = grad_w_bf16.to(torch.float32)

    return grad_x, grad_w


torch.library.register_autograd(
    "forgefuse::fused_mlp_up_relu2",
    _backward,
    setup_context=_setup_context,
)


def fused_mlp_up(x: Tensor, w: Tensor) -> Tensor:
    """Public API: relu(F.linear(x, w)).square(), fused into a single Triton kernel.

    Args:
        x: [..., K] input activations (bf16 or fp32; fp32 is cast to bf16 first
           to match F.linear-under-autocast semantics)
        w: [N, K] fp32 weight (CastedLinear stores fp32, casts at matmul time)
    Returns:
        [..., N] bf16, equal to `relu(x @ w.T).square()` within bf16 cast noise.
    """
    # Custom ops don't participate in PyTorch's autocast routing by default, so
    # a caller under `torch.autocast(bf16)` may still pass fp32 x (e.g. out of
    # F.rms_norm, which preserves input precision). Cast here to match what
    # F.linear-under-autocast would do for the reference path.
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    y_sq, _y = _fused_mlp_up_relu2_impl(x, w)
    return y_sq


# ============================================================================
# W8A8 REFERENCE IMPLEMENTATION (pure PyTorch)
# ============================================================================
# Path M3: fp32 master weights preserved (CastedLinear-native); per-forward
# fake-quant to INT8 for both weights and activations; STE backward.
#
# This is the mathematical equivalent of what a Triton INT8-tensor-core GEMM
# would produce (X_int8 @ W_int8^T -> int32 -> scale_x * scale_w * int32).
# We run the reference FIRST to verify training signal under pr-1493's Muon +
# AdamW + qk_gain_init=1.5 dynamics BEFORE investing in a Triton kernel.
#
# Why no torch.autograd.Function: phase3a's W8A8 used autograd.Function with
# requires_grad_() calls in backward which broke torch.compile(fullgraph=True).
# This reference uses pure PyTorch op composition — the `+ detach()` STE trick
# composes cleanly with dynamo tracing.
#
# See docs/research/compile_compatibility_rules.md Rule 1–3.

import torch.nn.functional as F  # noqa: E402  (late import, tied to W8A8 section)


def _quantize_to_int8_per_row(t: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric per-last-dim-row INT8 fake-quant, returns fp32 dequant value.

    t: fp32 tensor of any shape [*, K]
    Returns round(t/scale) * scale clipped to [-127, 127]*scale, same shape.
    """
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = amax / 127.0
    t_int = torch.round(t / scale).clamp(-127.0, 127.0)
    return t_int * scale


def _ste_fake_quant(t: Tensor) -> Tensor:
    """Straight-through estimator: forward is fake-quant, backward is identity."""
    t_q = _quantize_to_int8_per_row(t)
    return t + (t_q - t).detach()


def w8a8_linear_reference(x: Tensor, w: Tensor) -> Tensor:
    """Reference W8A8 linear: fake-quantizes x and w, fp32 matmul, STE backward.

    Equivalent to F.linear(fake_quant(x), fake_quant(w)) with gradients flowing
    through x and w as if quantization were identity.

    Args:
        x: [..., K] activations (any fp dtype; promoted to fp32 for quant math)
        w: [N, K] weight (any fp dtype; promoted to fp32 for quant math)
    Returns:
        [..., N] in x.dtype.
    """
    target_dtype = x.dtype
    x_q = _ste_fake_quant(x.float())
    w_q = _ste_fake_quant(w.float())
    # Cast to activation dtype for the matmul, matching CastedLinear/autocast.
    return F.linear(x_q.to(target_dtype), w_q.to(target_dtype))


def fused_mlp_up_w8a8_reference(x: Tensor, w: Tensor) -> Tensor:
    """Reference W8A8 variant of fused_mlp_up: W8A8(fc) + relu^2.

    Replaces fused_mlp_up when the W8A8 toggle is active. Does NOT quantize
    the relu^2 output — that's the next linear's input, and the next linear
    (proj) handles its own activation quant.
    """
    h = w8a8_linear_reference(x, w)
    return torch.relu(h).square()


# ============================================================================
# W8A8 TRITON KERNEL (Phase 4B) — INT8 tensor cores
# ============================================================================
# Replaces Phase 4A's fp32 reference with an actual INT8 GEMM. Uses the same
# triton_op + register_fake + register_autograd template as Kernel 2 (K2),
# proven compile-safe under fullgraph=True.
#
# Design (approved in Step 2 design report):
#   - Scales pre-computed in PyTorch ops inside the triton_op wrapper
#     (compiles cleanly under dynamo; verified via pre-check)
#   - Triton kernel consumes int8 + fp32 scales, returns bf16
#   - INT8 GEMM via tl.dot(int8, int8, out_dtype=int32)
#   - Dequant rule: int32 -> fp32 FIRST, then apply x_scale * w_scale as
#     separate multiplies (never precompute the product — underflow risk
#     when both scales are small, per phase3a W8A8 memory)
#   - Analytical STE backward: grad_x = grad_out @ q(w), grad_w = grad_out.T @ q(x)
#     (bf16 matmul for now; Phase 4C candidate if backward becomes bottleneck)

_W8A8_TRITON_CONFIGS = [
    # BLOCK_K >= 16 for H100/Blackwell INT8 tensor cores.
    triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_W8A8_TRITON_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def _int8_gemm_dequant_kernel(
    X_ptr, W_ptr, XS_ptr, WS_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """INT8 GEMM with per-row dequant.

    Inputs:
        X  [M, K] int8     (per-row quantized activations)
        W  [N, K] int8     (per-row quantized weight)
        XS [M]    fp32     (per-row activation scale)
        WS [N]    fp32     (per-row weight scale)
    Output:
        OUT [M, N] bf16, = (X @ W.T).to(fp32) * XS[:, None] * WS[None, :]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x_tile = tl.load(
            x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0
        )
        w_tile = tl.load(
            w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_rem), other=0
        )
        # INT8 TENSOR CORE MATMUL: x_tile [BLOCK_M, BLOCK_K] int8,
        # trans(w_tile) [BLOCK_K, BLOCK_N] int8, acc int32
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    x_scale = tl.load(XS_ptr + offs_m, mask=offs_m < M, other=1.0)
    w_scale = tl.load(WS_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Dequant: int32 -> fp32 FIRST, then scales as separate multiplies.
    # Avoid precomputing x_scale * w_scale — when both are small, product
    # underflows bf16 normals and truncates gradient magnitude.
    acc_fp32 = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :]
    out = acc_fp32.to(tl.bfloat16)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# NOTE: we use @torch.library.custom_op here, NOT @triton_op. Reason: Inductor
# tries to DECOMPOSE triton_op bodies into its own regenerated Triton source
# inside a temp module. That regeneration path fails for INT8 tl.dot (with
# out_dtype=int32) — the temp module can't resolve @jit's source file and
# raises "@jit functions should be defined in a Python file". custom_op is
# opaque to Inductor (treated as a black-box op), sidestepping the decomposition
# entirely. Same compile-safety outcome, different layering.
from torch.library import custom_op  # noqa: E402


@custom_op("forgefuse::w8a8_linear_forward", mutates_args=())
def _w8a8_linear_forward(x: Tensor, w: Tensor) -> Tensor:
    """Forward: quantize x and w to int8 (PyTorch ops), then INT8 GEMM+dequant (Triton)."""
    assert x.is_cuda and w.is_cuda
    assert w.dim() == 2, f"weight must be 2D, got {tuple(w.shape)}"
    orig_shape = x.shape
    K = orig_shape[-1]
    assert w.shape[-1] == K, f"K mismatch: x {K}, w {w.shape[-1]}"

    # Per-row INT8 quantization
    x_f32 = x.float()
    x_scale_keepdim = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    x_int8_full = (x_f32 / x_scale_keepdim).round().clamp(-127.0, 127.0).to(torch.int8)

    w_scale_keepdim = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    w_int8 = (w / w_scale_keepdim).round().clamp(-127.0, 127.0).to(torch.int8)

    x_int8_flat = x_int8_full.reshape(-1, K).contiguous()
    x_scale_flat = x_scale_keepdim.reshape(-1).contiguous()
    w_int8_c = w_int8.contiguous()
    w_scale_c = w_scale_keepdim.squeeze(-1).contiguous()

    M = x_int8_flat.shape[0]
    N = w_int8_c.shape[0]
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # Direct kernel call (not wrap_triton): custom_op is opaque to Inductor,
    # so we don't need the wrap_triton trace-visibility hook.
    _int8_gemm_dequant_kernel[grid](
        x_int8_flat, w_int8_c, x_scale_flat, w_scale_c, out_flat,
        M, N, K,
        x_int8_flat.stride(0), x_int8_flat.stride(1),
        w_int8_c.stride(0), w_int8_c.stride(1),
        out_flat.stride(0), out_flat.stride(1),
    )
    return out_flat.reshape(*orig_shape[:-1], N)


@_w8a8_linear_forward.register_fake
def _w8a8_linear_forward_fake(x, w):
    out_shape = list(x.shape[:-1]) + [w.shape[0]]
    return torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)


def _w8a8_setup_context(ctx, inputs, output):
    x, w = inputs
    ctx.save_for_backward(x, w)


def _w8a8_backward(ctx, grad_out):
    """Analytical STE backward.

    Forward (semantically): out = q(x) @ q(w).T where q = per-row INT8 fake-quant.
    STE: d q(t) / d t = 1 (identity), so:
        grad_x = grad_out @ q(w)       [..., N] @ [N, K] -> [..., K]
        grad_w = grad_out^T @ q(x)     [N, M] @ [M, K] -> [N, K]

    Pure functional: no requires_grad_(), no re-forward. bf16 matmul for speed.
    """
    x, w = ctx.saved_tensors
    with torch.no_grad():
        x_q_fp32 = _quantize_to_int8_per_row(x.float())
        w_q_fp32 = _quantize_to_int8_per_row(w.float())

    x_q_bf = x_q_fp32.to(torch.bfloat16)
    w_q_bf = w_q_fp32.to(torch.bfloat16)
    go = grad_out if grad_out.dtype == torch.bfloat16 else grad_out.to(torch.bfloat16)

    orig_x_shape = x.shape
    N = w.shape[0]
    K = w.shape[1]
    go_flat = go.reshape(-1, N)

    grad_x_flat = go_flat @ w_q_bf
    grad_x = grad_x_flat.reshape(orig_x_shape)
    grad_w = (go_flat.t() @ x_q_bf.reshape(-1, K)).to(torch.float32)
    return grad_x, grad_w


torch.library.register_autograd(
    "forgefuse::w8a8_linear_forward",
    _w8a8_backward,
    setup_context=_w8a8_setup_context,
)


def w8a8_linear_triton(x: Tensor, w: Tensor) -> Tensor:
    """Triton W8A8 linear (Phase 4B).

    Functionally equivalent to w8a8_linear_reference (Phase 4A) within INT8
    GEMM noise. Uses tl.dot(int8, int8, out_dtype=int32) for tensor-core
    throughput on H100 and INT8-capable Blackwell.

    Args:
        x: [..., K] activations (fp dtype; internally quantized)
        w: [N, K] fp32 weight (internally quantized)
    Returns:
        [..., N] bf16
    """
    return _w8a8_linear_forward(x, w)


def fused_mlp_up_w8a8_triton(x: Tensor, w: Tensor) -> Tensor:
    """Triton W8A8 variant of fused_mlp_up: W8A8(fc) + relu^2."""
    h = w8a8_linear_triton(x, w)
    return torch.relu(h).square()
