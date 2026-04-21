"""
Fused Attention Preprocessing for Parameter Golf.

Two Triton kernels:
1. fused_qk_norm_gain: Per-head RMSNorm + optional gain (Q only) — replaces 3 kernels
2. fused_partial_rope: In-place partial RoPE on first 16/64 dims — replaces 2 kernels

Combined with the autograd function that shares the input RMSNorm across Q/K/V projections,
this eliminates 8+ kernel launches from the attention preprocessing path.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl

HEAD_DIM: int = 64
ROPE_DIMS: int = 16


# ============================================================
# Kernel 1: Per-head RMSNorm + optional gain
# ============================================================

@triton.jit
def fused_qk_norm_gain_kernel(
    X_ptr,        # [M, num_heads, HD] — modified in-place
    GAIN_ptr,     # [num_heads] or unused
    M,
    NUM_HEADS: tl.constexpr,
    HD: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    stride_m, stride_h, stride_d,
    BLOCK_M: tl.constexpr,
):
    """Per-head RMSNorm, optionally multiplied by per-head gain."""
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    d_offs = tl.arange(0, HD)

    for h in range(NUM_HEADS):
        ptrs = X_ptr + m_offs[:, None] * stride_m + h * stride_h + d_offs[None, :] * stride_d
        head = tl.load(ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # RMSNorm
        var = tl.sum(head * head, axis=1) / HD
        head = head * tl.rsqrt(var[:, None] + 1e-6)

        # Optional gain
        if HAS_GAIN:
            gain = tl.load(GAIN_ptr + h).to(tl.float32)
            head = head * gain

        tl.store(ptrs, head.to(tl.bfloat16), mask=m_mask[:, None])


# ============================================================
# Kernel 2: In-place partial RoPE
# ============================================================

@triton.jit
def fused_partial_rope_kernel(
    X_ptr,        # [M, num_heads, HD] — modified in-place (first ROPE_DIMS dims)
    COS_ptr,      # [S, rope_half]
    SIN_ptr,      # [S, rope_half]
    M, S,
    NUM_HEADS: tl.constexpr,
    HD: tl.constexpr,
    ROPE_HALF: tl.constexpr,
    stride_m, stride_h, stride_d,
    stride_cos_s, stride_cos_r,
    BLOCK_M: tl.constexpr,
):
    """Apply partial RoPE: rotate first ROPE_HALF*2 dims, leave rest unchanged."""
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    pos = m_offs % S
    r_offs = tl.arange(0, ROPE_HALF)

    # Load cos/sin for this position block
    cos_ptrs = COS_ptr + pos[:, None] * stride_cos_s + r_offs[None, :] * stride_cos_r
    sin_ptrs = SIN_ptr + pos[:, None] * stride_cos_s + r_offs[None, :] * stride_cos_r
    cos_val = tl.load(cos_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    for h in range(NUM_HEADS):
        base = X_ptr + m_offs[:, None] * stride_m + h * stride_h

        # Load first half (dims 0..ROPE_HALF-1)
        r1_ptrs = base + r_offs[None, :] * stride_d
        x1 = tl.load(r1_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Load second half (dims ROPE_HALF..2*ROPE_HALF-1)
        r2_ptrs = base + (ROPE_HALF + r_offs)[None, :] * stride_d
        x2 = tl.load(r2_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Rotate
        out1 = x1 * cos_val + x2 * sin_val
        out2 = x1 * (-sin_val) + x2 * cos_val

        tl.store(r1_ptrs, out1.to(tl.bfloat16), mask=m_mask[:, None])
        tl.store(r2_ptrs, out2.to(tl.bfloat16), mask=m_mask[:, None])


# ============================================================
# Python wrappers
# ============================================================

def fused_qk_norm_gain(x, gain=None):
    """In-place per-head RMSNorm + optional gain. x: [B, S, H, hd]."""
    B, S, NH, HD = x.shape
    M = B * S
    x_flat = x.reshape(M, NH, HD)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M),)
    has_gain = gain is not None
    fused_qk_norm_gain_kernel[grid](
        x_flat, gain.float() if has_gain else x_flat,
        M, NH, HD, has_gain,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        BLOCK_M=BLOCK_M,
    )
    return x


def fused_partial_rope(x, cos, sin, seq_len):
    """In-place partial RoPE. x: [B, S, H, hd], cos/sin: [1, S, 1, rope_half]."""
    B, S, NH, HD = x.shape
    M = B * S
    ROPE_HALF = ROPE_DIMS // 2
    x_flat = x.reshape(M, NH, HD)
    cos_flat = cos.squeeze(0).squeeze(-2).contiguous().to(torch.bfloat16)
    sin_flat = sin.squeeze(0).squeeze(-2).contiguous().to(torch.bfloat16)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M),)
    fused_partial_rope_kernel[grid](
        x_flat, cos_flat, sin_flat,
        M, seq_len, NH, HD, ROPE_HALF,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        cos_flat.stride(0), cos_flat.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return x


# ============================================================
# Autograd function: full preprocessing chain
# ============================================================

class FusedAttnPreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, q_w, k_w, v_w, q_gain, cos, sin, ln_scale, num_heads, num_kv_heads, v_embed=None):
        B, S, D = x_in.shape
        hd = D // num_heads

        # Shared RMSNorm (one pass)
        x_normed = F.rms_norm(x_in, (D,)) * ln_scale

        # Projections (cuBLAS)
        q = F.linear(x_normed, q_w.to(x_normed.dtype)).reshape(B, S, num_heads, hd)
        k = F.linear(x_normed, k_w.to(x_normed.dtype)).reshape(B, S, num_kv_heads, hd)
        v = F.linear(x_normed, v_w.to(x_normed.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, S, num_kv_heads, hd)

        # Fused QK postprocessing
        q = fused_qk_norm_gain(q.contiguous(), gain=q_gain)
        fused_partial_rope(q, cos, sin, S)
        k = fused_qk_norm_gain(k.contiguous(), gain=None)
        fused_partial_rope(k, cos, sin, S)

        ctx.save_for_backward(x_in, q_w, k_w, v_w, q_gain, cos, sin)
        ctx.ln_scale = ln_scale
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.has_ve = v_embed is not None
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        x_in, q_w, k_w, v_w, q_gain, cos, sin = ctx.saved_tensors
        ln_scale = ctx.ln_scale
        B, S, D = x_in.shape
        M = B * S
        hd = D // ctx.num_heads
        rope_half = ROPE_DIMS // 2

        x_flat = x_in.reshape(M, D).float()
        variance = (x_flat ** 2).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + 1e-6)
        x_normed = x_flat * rms_inv * ln_scale

        q_proj = F.linear(x_normed.to(q_w.dtype), q_w).reshape(M, ctx.num_heads, hd)
        k_proj = F.linear(x_normed.to(k_w.dtype), k_w).reshape(M, ctx.num_kv_heads, hd)
        q_n = F.rms_norm(q_proj.float(), (hd,))
        k_n = F.rms_norm(k_proj.float(), (hd,))

        cos_r = cos.squeeze(0).squeeze(-2).float()
        sin_r = sin.squeeze(0).squeeze(-2).float()
        pos = torch.arange(M, device=x_in.device) % S
        c = cos_r[pos].unsqueeze(1)
        s = sin_r[pos].unsqueeze(1)

        def rope_fwd(x):
            x1, x2, xp = x[..., :rope_half], x[..., rope_half:ROPE_DIMS], x[..., ROPE_DIMS:]
            return torch.cat([x1*c+x2*s, x1*(-s)+x2*c, xp], -1)

        def rope_bwd(g):
            g1, g2, gp = g[..., :rope_half], g[..., rope_half:ROPE_DIMS], g[..., ROPE_DIMS:]
            return torch.cat([g1*c+g2*(-s), g1*s+g2*c, gp], -1)

        q_roped = rope_fwd(q_n)
        gq = grad_q.reshape(M, ctx.num_heads, hd).float()
        gk = grad_k.reshape(M, ctx.num_kv_heads, hd).float()

        grad_q_gain = (gq * q_roped).sum(dim=(0, 2))
        gq_pre = gq * q_gain.float().unsqueeze(0).unsqueeze(-1)
        gq_pre_rope = rope_bwd(gq_pre)
        gk_pre_rope = rope_bwd(gk)

        def rms_bwd(g, x, d):
            v = (x**2).mean(-1, keepdim=True)
            r = torch.rsqrt(v + 1e-6)
            return g * r - x * (g * x).sum(-1, keepdim=True) * (r**3) / d

        gq_proj = rms_bwd(gq_pre_rope, q_proj.float(), hd).reshape(M, -1)
        gk_proj = rms_bwd(gk_pre_rope, k_proj.float(), hd).reshape(M, -1)
        gv_proj = grad_v.reshape(M, -1).float()

        gx_q = F.linear(gq_proj.to(q_w.dtype), q_w.t())
        gx_k = F.linear(gk_proj.to(k_w.dtype), k_w.t())
        gx_v = F.linear(gv_proj.to(v_w.dtype), v_w.t())
        g_qw = gq_proj.t().to(x_normed.dtype) @ x_normed.to(gq_proj.dtype)
        g_kw = gk_proj.t().to(x_normed.dtype) @ x_normed.to(gk_proj.dtype)
        g_vw = gv_proj.t().to(x_normed.dtype) @ x_normed.to(gv_proj.dtype)

        gxn = (gx_q + gx_k + gx_v).float() * ln_scale
        dx = gxn * rms_inv - x_flat * (gxn * x_flat).sum(-1, keepdim=True) * (rms_inv**3) / D

        return dx.to(x_in.dtype).reshape(B,S,D), g_qw.to(q_w.dtype), g_kw.to(k_w.dtype), g_vw.to(v_w.dtype), grad_q_gain, None, None, None, None, None, None


def fused_attn_pre(x_in, q_w, k_w, v_w, q_gain, cos, sin, ln_scale, num_heads, num_kv_heads, v_embed=None, use_triton=True):
    if use_triton and x_in.is_cuda:
        return FusedAttnPreFunction.apply(x_in, q_w, k_w, v_w, q_gain, cos, sin, ln_scale, num_heads, num_kv_heads, v_embed)
    B, S, D = x_in.shape
    hd = D // num_heads
    x_n = F.rms_norm(x_in, (D,)) * ln_scale
    q = F.linear(x_n, q_w.to(x_n.dtype)).reshape(B,S,num_heads,hd)
    k = F.linear(x_n, k_w.to(x_n.dtype)).reshape(B,S,num_kv_heads,hd)
    v = F.linear(x_n, v_w.to(x_n.dtype))
    if v_embed is not None:
        v = v + v_embed
    v = v.reshape(B,S,num_kv_heads,hd)
    q = F.rms_norm(q, (hd,))
    k = F.rms_norm(k, (hd,))
    c, s = cos.to(q.dtype), sin.to(q.dtype)
    def rope(x):
        half = ROPE_DIMS // 2
        x1, x2, xp = x[..., :half], x[..., half:ROPE_DIMS], x[..., ROPE_DIMS:]
        return torch.cat((x1*c+x2*s, x1*(-s)+x2*c, xp), -1)
    q, k = rope(q), rope(k)
    q = q * q_gain.to(q.dtype)[None, None, :, None]
    return q, k, v


def test():
    torch.manual_seed(42)
    B, S, D, NH, NKV = 2, 64, 512, 8, 4
    ln = 1.0/3**0.5
    x = torch.randn(B,S,D, device='cuda', dtype=torch.bfloat16)
    qw = torch.randn(D,D, device='cuda', dtype=torch.float32)
    kw = torch.randn(NKV*64,D, device='cuda', dtype=torch.float32)
    vw = torch.randn(NKV*64,D, device='cuda', dtype=torch.float32)
    qg = torch.full((NH,), 1.5, device='cuda', dtype=torch.float32)
    inv = 1.0/(10000**(torch.arange(0,ROPE_DIMS,2,device='cuda',dtype=torch.float32)/ROPE_DIMS))
    t = torch.arange(S,device='cuda',dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos = freqs.cos()[None,:,None,:]
    sin = freqs.sin()[None,:,None,:]

    qr,kr,vr = fused_attn_pre(x,qw,kw,vw,qg,cos,sin,ln,NH,NKV, use_triton=False)
    qt,kt,vt = fused_attn_pre(x,qw,kw,vw,qg,cos,sin,ln,NH,NKV, use_triton=True)
    qc = F.cosine_similarity(qr.float().reshape(1,-1), qt.float().reshape(1,-1)).item()
    kc = F.cosine_similarity(kr.float().reshape(1,-1), kt.float().reshape(1,-1)).item()
    vc = F.cosine_similarity(vr.float().reshape(1,-1), vt.float().reshape(1,-1)).item()
    print(f"Q cos={qc:.8f}  K cos={kc:.8f}  V cos={vc:.8f}")
    assert qc > 0.999 and kc > 0.999 and vc > 0.999
    print("PASSED!")


if __name__ == "__main__":
    test()
