"""
Kernel 3: Fused QK Norm + Partial RoPE + Q Gain

Fuses 5 element-wise ops between QKV GEMMs and Flash Attention:
  1. RMSNorm(q, dim=64)
  2. RMSNorm(k, dim=64)
  3. Partial RoPE on q (dims 0-15 only, 16-63 passthrough)
  4. Partial RoPE on k (dims 0-15 only, 16-63 passthrough)
  5. q *= q_gain (per-head scalar)

One kernel function handles both Q (HAS_GAIN=True) and K (HAS_GAIN=False).
Each program processes one (batch, token, head) tuple — all 64 dims in registers.
"""
import os
os.environ.setdefault('CC', r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe')

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

_USE_TRITON_ATTN = True
ROPE_DIMS: int = 16  # partial RoPE: only first 16 of 64 dims
HEAD_DIM: int = 64

@triton.jit
def _fused_qk_norm_rope_kernel(
    # Pointers
    x_ptr,          # [B, T, H, D]  input/output (in-place)
    cos_ptr,        # [1, T, 1, ROPE_HALF]  RoPE cosines
    sin_ptr,        # [1, T, 1, ROPE_HALF]  RoPE sines
    gain_ptr,       # [H]  per-head gain (NULL if HAS_GAIN=False)
    # Dimensions
    B, T, H,
    # Strides for x: [B, T, H, D]
    stride_xb, stride_xt, stride_xh, stride_xd,
    # Strides for cos/sin: [1, T, 1, ROPE_HALF]
    stride_cos_t,
    # Constants
    D: tl.constexpr,           # head_dim = 64
    ROPE_HALF: tl.constexpr,   # rope_dims // 2 = 8
    HAS_GAIN: tl.constexpr,    # True for Q, False for K
    EPS: tl.constexpr,         # RMSNorm epsilon = 1e-6
):
    """Fused RMSNorm + Partial RoPE + optional Gain for one (b, t, h) tuple."""
    # Each program handles one (batch, token, head)
    pid = tl.program_id(0)
    total_bth = B * T * H
    if pid >= total_bth:
        return

    # Decompose pid -> (b, t, h)
    b = pid // (T * H)
    rem = pid % (T * H)
    t = rem // H
    h = rem % H

    # -- Step 1: Load x[b, t, h, 0:D] into registers (bf16 → f32) --
    base = b * stride_xb + t * stride_xt + h * stride_xh
    offs_d = tl.arange(0, D)
    x = tl.load(x_ptr + base + offs_d * stride_xd).to(tl.float32)

    # -- Step 2: RMSNorm (no learned weight) --
    var = tl.sum(x * x, axis=0) / D
    x = x * tl.rsqrt(var + EPS)

    # -- Step 3: Partial RoPE on dims 0..2*ROPE_HALF-1 only --
    # Load cos/sin for this token: cos_ptr[0, t, 0, 0:ROPE_HALF]
    cos_base = t * stride_cos_t
    cos_offs = tl.arange(0, ROPE_HALF)
    cos_vals = tl.load(cos_ptr + cos_base + cos_offs).to(tl.float32)
    sin_vals = tl.load(sin_ptr + cos_base + cos_offs).to(tl.float32)

    # x1 = x[0:ROPE_HALF], x2 = x[ROPE_HALF:2*ROPE_HALF]
    # Mask-based approach: only modify dims 0..2*ROPE_HALF-1
    x1_mask = offs_d < ROPE_HALF                    # dims 0..7
    x2_mask = (offs_d >= ROPE_HALF) & (offs_d < 2 * ROPE_HALF)  # dims 8..15

    # Extract x1 and x2 values at their positions
    # For x1 positions (0..7): new = x1*cos + x2*sin
    # For x2 positions (8..15): new = -x1*sin + x2*cos
    # For passthrough (16..63): unchanged

    # Gather x1 and x2 values
    x1_vals = tl.load(x_ptr + base + tl.arange(0, ROPE_HALF) * stride_xd).to(tl.float32)
    x2_vals = tl.load(x_ptr + base + (tl.arange(0, ROPE_HALF) + ROPE_HALF) * stride_xd).to(tl.float32)

    # But we already have x normalized — x1/x2 should come from the normalized x, not raw load
    # Actually we modified x in-place (in register). Let me use indexing on x directly.
    # Since D=64 is a constexpr and small, we can just work with the full vector.

    # Re-extract from the normalized x register vector
    # x is shape [D] in registers
    # x1 = x[0:8], x2 = x[8:16]
    # Problem: tl doesn't support slicing a 1D register vector by index range easily
    # Solution: use tl.where with masks to construct the rotated version

    # Build new_x1 = x[0:8]*cos - x[8:16]*sin   (note: original code has +sin, let me check)
    # Original: x1*cos + x2*sin,  x1*(-sin) + x2*cos
    # So: new_x1 = x1*cos + x2*sin
    #     new_x2 = -x1*sin + x2*cos

    # We need to scatter these back. Since D is small and constexpr,
    # build the full output vector with tl.where:

    # For each dim d:
    #   if d < ROPE_HALF:       out[d] = x[d]*cos[d] + x[d+ROPE_HALF]*sin[d]
    #   if ROPE_HALF <= d < 2*ROPE_HALF: out[d] = -x[d-ROPE_HALF]*sin[d-ROPE_HALF] + x[d]*cos[d-ROPE_HALF]
    #   else:                   out[d] = x[d]

    # We need cos/sin indexed by (d % ROPE_HALF) for dims in range
    # And we need x values from paired dims

    # Simpler: just build two separate result segments and merge
    # Since we can't slice register tensors, let's use gather/scatter with full-width ops

    # Approach: create rope_cos and rope_sin expanded to full D width
    # Only first 2*ROPE_HALF dims are affected
    # For dim d < ROPE_HALF: cos_full[d] = cos[d], sin_full[d] = sin[d]
    # For dim ROPE_HALF <= d < 2*ROPE_HALF: cos_full[d] = cos[d-ROPE_HALF], sin_full[d] = sin[d-ROPE_HALF]

    # Partner dim: for d in [0, ROPE_HALF): partner = d + ROPE_HALF
    #              for d in [ROPE_HALF, 2*ROPE_HALF): partner = d - ROPE_HALF
    partner = tl.where(offs_d < ROPE_HALF, offs_d + ROPE_HALF, offs_d - ROPE_HALF)
    # Clamp partner for dims >= 2*ROPE_HALF (they won't be used but need valid index)
    partner = tl.where(offs_d < 2 * ROPE_HALF, partner, offs_d)

    # Gather partner values from x
    # x_partner[d] = x[partner[d]]
    x_partner = tl.load(x_ptr + base + partner * stride_xd).to(tl.float32)
    # Recompute x_partner from the NORMALIZED x, not from memory
    # Actually x in registers IS normalized. We need x_partner from registers too.
    # Problem: tl doesn't support gather from register tensor.
    # We must re-derive from memory or restructure.

    # CLEAN APPROACH: Just do the RoPE with explicit loads of the 8+8 dims
    # We already have x (normalized, in f32 registers, shape [D])
    # We can store x back to memory, then load the specific dims.
    # OR: we compute the rope in a simpler way.

    # Let's store normalized x temporarily, then apply RoPE via targeted loads
    # This is one extra store+load, but it's to the same cache line.
    tl.store(x_ptr + base + offs_d * stride_xd, x.to(tl.bfloat16))

    # Now reload x1 (dims 0-7) and x2 (dims 8-15) from the normalized output
    rope_offs = tl.arange(0, ROPE_HALF)
    x1 = tl.load(x_ptr + base + rope_offs * stride_xd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (rope_offs + ROPE_HALF) * stride_xd).to(tl.float32)

    # Apply rotation
    new_x1 = x1 * cos_vals + x2 * sin_vals
    new_x2 = -x1 * sin_vals + x2 * cos_vals

    # Store rotated dims back (dims 0-7 and 8-15 only; 16-63 already stored correctly)
    tl.store(x_ptr + base + rope_offs * stride_xd, new_x1.to(tl.bfloat16))
    tl.store(x_ptr + base + (rope_offs + ROPE_HALF) * stride_xd, new_x2.to(tl.bfloat16))

    # -- Step 4: Gain (Q only) --
    if HAS_GAIN:
        gain_val = tl.load(gain_ptr + h).to(tl.float32)
        # Reload all D dims, multiply by gain, store back
        x_final = tl.load(x_ptr + base + offs_d * stride_xd).to(tl.float32)
        x_final = x_final * gain_val
        tl.store(x_ptr + base + offs_d * stride_xd, x_final.to(tl.bfloat16))


def _triton_qk_norm_rope_fwd(x, cos, sin, gain, rope_dims, has_gain):
    """Raw Triton forward for Q or K norm+rope+gain."""
    x = x.contiguous().clone().to(torch.bfloat16)
    B, T, H, D = x.shape
    ROPE_HALF = rope_dims // 2
    grid = (B * T * H,)
    gain_ptr = gain if has_gain else x  # dummy pointer when no gain
    _fused_qk_norm_rope_kernel[grid](
        x, cos.to(torch.bfloat16), sin.to(torch.bfloat16), gain_ptr,
        B, T, H,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(1),
        D=D, ROPE_HALF=ROPE_HALF, HAS_GAIN=has_gain, EPS=1e-6,
    )
    return x

def _pytorch_qk_norm_rope(x, cos, sin, gain, rope_dims, has_gain):
    """Reference PyTorch implementation for forward and backward."""
    x = F.rms_norm(x, (x.size(-1),))
    half = rope_dims // 2
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    x = torch.cat((x_rope, x_pass), dim=-1)
    if has_gain:
        x = x * gain[None, None, :, None]
    return x

class _FusedQKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, gain, rope_dims, has_gain):
        ctx.save_for_backward(x, cos, sin, gain)
        ctx.rope_dims = rope_dims
        ctx.has_gain = has_gain
        return _triton_qk_norm_rope_fwd(x, cos, sin, gain, rope_dims, has_gain)

    @staticmethod
    def backward(ctx, grad_output):
        x, cos, sin, gain = ctx.saved_tensors
        # Recompute forward with PyTorch ops (creates autograd graph for backward)
        with torch.enable_grad():
            x_det = x.detach().requires_grad_(True)
            out = _pytorch_qk_norm_rope(x_det, cos, sin, gain, ctx.rope_dims, ctx.has_gain)
            out.backward(grad_output)
        return x_det.grad, None, None, None, None, None


def fused_q_norm_rope_gain(q: Tensor, cos: Tensor, sin: Tensor, gain: Tensor,
                           rope_dims: int = ROPE_DIMS) -> Tensor:
    """Fused RMSNorm + partial RoPE + per-head gain for Q tensor."""
    if not _USE_TRITON_ATTN:
        return _pytorch_qk_norm_rope(q, cos, sin, gain, rope_dims, has_gain=True)
    return _FusedQKNormRoPE.apply(q, cos, sin, gain, rope_dims, True)


def fused_k_norm_rope(k: Tensor, cos: Tensor, sin: Tensor,
                      rope_dims: int = ROPE_DIMS) -> Tensor:
    """Fused RMSNorm + partial RoPE for K tensor (no gain)."""
    if not _USE_TRITON_ATTN:
        return _pytorch_qk_norm_rope(k, cos, sin, k, rope_dims, has_gain=False)
    return _FusedQKNormRoPE.apply(k, cos, sin, k, rope_dims, False)


# --- Correctness Test -------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(1337)
    device = 'cuda'

    B, T, H_q, H_k, D = 4, 1024, 8, 4, 64
    rope_dims = 16
    ROPE_HALF = rope_dims // 2

    q = torch.randn(B, T, H_q, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H_k, D, device=device, dtype=torch.bfloat16)
    cos = torch.randn(1, T, 1, ROPE_HALF, device=device, dtype=torch.bfloat16)
    sin = torch.randn(1, T, 1, ROPE_HALF, device=device, dtype=torch.bfloat16)
    gain = torch.randn(H_q, device=device, dtype=torch.bfloat16)

    print("=" * 60)
    print("Kernel 3: Fused QK Norm + Partial RoPE + Gain")
    print("=" * 60)

    # -- Reference (fallback path) --
    _USE_TRITON_ATTN_SAVE = _USE_TRITON_ATTN
    globals()['_USE_TRITON_ATTN'] = False

    q_ref = fused_q_norm_rope_gain(q.clone(), cos, sin, gain, rope_dims)
    k_ref = fused_k_norm_rope(k.clone(), cos, sin, rope_dims)

    globals()['_USE_TRITON_ATTN'] = True

    q_tri = fused_q_norm_rope_gain(q.clone(), cos, sin, gain, rope_dims)
    k_tri = fused_k_norm_rope(k.clone(), cos, sin, rope_dims)

    globals()['_USE_TRITON_ATTN'] = _USE_TRITON_ATTN_SAVE

    # -- Q correctness --
    cos_q = F.cosine_similarity(q_ref.flatten().float(), q_tri.flatten().float(), dim=0).item()
    diff_q = (q_ref.float() - q_tri.float()).abs().max().item()
    rel_q = ((q_ref.float() - q_tri.float()).abs() / (q_ref.float().abs() + 1e-8)).max().item()

    # Check passthrough dims (16-63) specifically
    cos_q_pass = F.cosine_similarity(
        q_ref[..., rope_dims:].flatten().float(),
        q_tri[..., rope_dims:].flatten().float(), dim=0
    ).item()

    print(f"\nQ tensor [{B},{T},{H_q},{D}]:")
    print(f"  cos_sim (all dims):     {cos_q:.6f}")
    print(f"  cos_sim (dims 16-63):   {cos_q_pass:.6f}  (passthrough check)")
    print(f"  max_abs_diff:           {diff_q:.6f}")
    print(f"  max_rel_err:            {rel_q:.6f}")

    # -- K correctness --
    cos_k = F.cosine_similarity(k_ref.flatten().float(), k_tri.flatten().float(), dim=0).item()
    diff_k = (k_ref.float() - k_tri.float()).abs().max().item()
    rel_k = ((k_ref.float() - k_tri.float()).abs() / (k_ref.float().abs() + 1e-8)).max().item()

    cos_k_pass = F.cosine_similarity(
        k_ref[..., rope_dims:].flatten().float(),
        k_tri[..., rope_dims:].flatten().float(), dim=0
    ).item()

    print(f"\nK tensor [{B},{T},{H_k},{D}]:")
    print(f"  cos_sim (all dims):     {cos_k:.6f}")
    print(f"  cos_sim (dims 16-63):   {cos_k_pass:.6f}  (passthrough check)")
    print(f"  max_abs_diff:           {diff_k:.6f}")
    print(f"  max_rel_err:            {rel_k:.6f}")

    # -- Verdict --
    q_pass = cos_q > 0.9999 and cos_q_pass > 0.9999
    k_pass = cos_k > 0.9999 and cos_k_pass > 0.9999
    print(f"\nQ: {'PASS' if q_pass else 'FAIL'}  |  K: {'PASS' if k_pass else 'FAIL'}")

    if not q_pass or not k_pass:
        print("\nDEBUG: checking dims individually...")
        for label, ref_t, tri_t in [("Q", q_ref, q_tri), ("K", k_ref, k_tri)]:
            for start, end, name in [(0, 8, "rope_x1(0-7)"), (8, 16, "rope_x2(8-15)"), (16, 64, "pass(16-63)")]:
                cs = F.cosine_similarity(
                    ref_t[..., start:end].flatten().float(),
                    tri_t[..., start:end].flatten().float(), dim=0
                ).item()
                md = (ref_t[..., start:end].float() - tri_t[..., start:end].float()).abs().max().item()
                print(f"  {label} dims {name}: cos={cs:.6f} max_diff={md:.6f}")
        sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════
    # STEP 3: BENCHMARK
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3: BENCHMARK (torch.cuda.Event, 500 iters, 20 warmup)")
    print("=" * 60)

    WARMUP, ITERS = 20, 500

    def bench(fn, warmup=WARMUP, iters=ITERS):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    # Baseline: separate ops
    def baseline_q():
        qq = F.rms_norm(q.clone(), (D,))
        half = rope_dims // 2
        xr, xp = qq[..., :rope_dims], qq[..., rope_dims:]
        x1, x2 = xr[..., :half], xr[..., half:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        qq = torch.cat((xr, xp), dim=-1)
        return qq * gain[None, None, :, None]

    def baseline_k():
        kk = F.rms_norm(k.clone(), (D,))
        half = rope_dims // 2
        xr, xp = kk[..., :rope_dims], kk[..., rope_dims:]
        x1, x2 = xr[..., :half], xr[..., half:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((xr, xp), dim=-1)

    def fused_q():
        return fused_q_norm_rope_gain(q.clone(), cos, sin, gain, rope_dims)

    def fused_k():
        return fused_k_norm_rope(k.clone(), cos, sin, rope_dims)

    ms_base_q = bench(baseline_q)
    ms_base_k = bench(baseline_k)
    ms_fused_q = bench(fused_q)
    ms_fused_k = bench(fused_k)

    ms_base_total = ms_base_q + ms_base_k
    ms_fused_total = ms_fused_q + ms_fused_k
    attn_speedup = ms_base_total / ms_fused_total
    attn_save_qk = ms_base_total - ms_fused_total

    print(f"\n  Separate ops (baseline):")
    print(f"    Q (norm+rope+gain):   {ms_base_q:.4f} ms")
    print(f"    K (norm+rope):        {ms_base_k:.4f} ms")
    print(f"    Total Q+K:            {ms_base_total:.4f} ms")
    print(f"\n  Triton fused:")
    print(f"    Q (fused):            {ms_fused_q:.4f} ms")
    print(f"    K (fused):            {ms_fused_k:.4f} ms")
    print(f"    Total Q+K:            {ms_fused_total:.4f} ms")
    print(f"\n  Speedup:                {attn_speedup:.2f}x")
    print(f"  Savings per attn block: {attn_save_qk:.4f} ms")
    print(f"  Savings per step (x11): {attn_save_qk * 11:.2f} ms")

    # ═════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY (MLP + Attention)
    # ═════════════════════════════════════════════════════════════════════
    # Import MLP savings from the previous validation (hardcode the measured values)
    # Re-measure MLP here for consistency
    import train_gpt
    from train_gpt import _triton_mlp_up_proj_fwd, _triton_mlp_down_proj_fwd

    M_mlp, K_up, N_up = 4096, 512, 1536  # up: [M,512] @ [512,1536]
    K_dn, N_dn = 1536, 512               # down: [M,1536] @ [1536,512]
    x_mlp = torch.randn(M_mlp, K_up, device=device, dtype=torch.bfloat16)
    w_up = torch.randn(N_up, K_up, device=device, dtype=torch.bfloat16)  # [1536, 512]
    w_dn = torch.randn(N_dn, K_dn, device=device, dtype=torch.bfloat16)  # [512, 1536]
    sc = torch.randn(N_dn, device=device, dtype=torch.bfloat16)  # [512]
    res = torch.randn(M_mlp, N_dn, device=device, dtype=torch.bfloat16)  # [M, 512]
    x_dn = torch.randn(M_mlp, K_dn, device=device, dtype=torch.bfloat16)  # [M, 1536]

    ms_cb_up = bench(lambda: F.leaky_relu(F.linear(x_mlp, w_up), negative_slope=0.5).square())
    ms_tr_up = bench(lambda: _triton_mlp_up_proj_fwd(x_mlp, w_up))
    ms_cb_dn = bench(lambda: res + sc[None, :] * F.linear(x_dn, w_dn))
    ms_tr_dn = bench(lambda: _triton_mlp_down_proj_fwd(x_dn, w_dn, sc, res))

    save_up = ms_cb_up - ms_tr_up
    save_dn = ms_cb_dn - ms_tr_dn

    print("\n" + "=" * 60)
    print("COMBINED SAVINGS PER STEP (MLP + Attention, 11 blocks)")
    print("=" * 60)
    print(f"  MLP up proj (x11):    {save_up:.4f} ms x 11 = {save_up*11:.2f} ms")
    print(f"  MLP down proj (x11):  {save_dn:.4f} ms x 11 = {save_dn*11:.2f} ms")
    print(f"  Attn Q+K (x11):       {attn_save_qk:.4f} ms x 11 = {attn_save_qk*11:.2f} ms")
    total_save = (save_up + save_dn + attn_save_qk) * 11
    print(f"  {'-' * 40}")
    print(f"  Total per step:       {total_save:.2f} ms")
    print(f"  vs 86.7ms baseline:   {total_save/86.7*100:.1f}%")

    # ═════════════════════════════════════════════════════════════════════
    # 200-STEP LOSS DIFF VALIDATION
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("200-STEP LOSS DIFF (attn fusion ON vs OFF)")
    print("=" * 60)

    import math as _math
    from torch import nn

    MODEL_DIM, MLP_DIM_V = 512, 1536
    Bv, Tv = 2, 128
    N_STEPS = 200
    checkpoints = {1, 10, 50, 100, 200}

    class AttnMLPBlock(nn.Module):
        """Isolated block: tests both MLP and attention norm/rope/gain paths."""
        def __init__(self):
            super().__init__()
            self.q_w = nn.Parameter(torch.randn(MODEL_DIM, MODEL_DIM) * 0.02)
            self.k_w = nn.Parameter(torch.randn(MODEL_DIM // 2, MODEL_DIM) * 0.02)
            self.q_gain = nn.Parameter(torch.full((8,), 1.5))
            self.rotary = train_gpt.Rotary(64, rope_dims=16)
            self.num_heads = 8
            self.num_kv_heads = 4
            self.head_dim = 64
            self.rope_dims = 16

        def forward(self, x):
            bsz, seqlen, dim = x.shape
            q = F.linear(x, self.q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
            k = F.linear(x, self.k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            cos, sin_v = self.rotary(seqlen, x.device, q.dtype)

            if _USE_TRITON_ATTN:
                q = fused_q_norm_rope_gain(q, cos, sin_v, self.q_gain.to(q.dtype), self.rope_dims)
                k = fused_k_norm_rope(k, cos, sin_v, self.rope_dims)
            else:
                q = F.rms_norm(q, (q.size(-1),))
                k = F.rms_norm(k, (k.size(-1),))
                q = train_gpt.apply_rotary_emb(q, cos, sin_v, self.rope_dims)
                k = train_gpt.apply_rotary_emb(k, cos, sin_v, self.rope_dims)
                q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

            # Simple output: flatten Q heads as proxy
            q_flat = q.reshape(bsz, seqlen, -1)[..., :MODEL_DIM]  # [B,T,512]
            k_flat = k.reshape(bsz, seqlen, -1)  # [B,T,256]
            return q_flat + k_flat.repeat(1, 1, 2)  # broadcast K to [B,T,512]

    losses_base = {}
    losses_fused = {}

    for mode_name, use_tri in [("baseline", False), ("fused", True)]:
        globals()['_USE_TRITON_ATTN'] = use_tri
        torch.manual_seed(42)
        model = AttnMLPBlock().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        store = losses_base if mode_name == "baseline" else losses_fused

        for step in range(1, N_STEPS + 1):
            torch.manual_seed(1000 + step)
            x = torch.randn(Bv, Tv, MODEL_DIM, device=device, dtype=torch.bfloat16)
            target = torch.randn(Bv, Tv, MODEL_DIM, device=device, dtype=torch.bfloat16)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = F.mse_loss(out.float(), target.float())
            loss.backward()
            opt.step()
            if step in checkpoints:
                store[step] = loss.item()

        del model, opt
        torch.cuda.empty_cache()

    globals()['_USE_TRITON_ATTN'] = True

    print(f"\n{'Step':>6} | {'Loss (baseline)':>15} | {'Loss (fused)':>14} | {'Diff':>10}")
    print("-" * 56)
    max_loss_diff = 0.0
    for step in sorted(checkpoints):
        lb = losses_base[step]
        lf = losses_fused[step]
        d = abs(lb - lf)
        max_loss_diff = max(max_loss_diff, d)
        print(f"{step:>6} | {lb:>15.6f} | {lf:>14.6f} | {d:>10.6f}")
    print(f"\nMax loss diff: {max_loss_diff:.6f}")
    loss_pass = max_loss_diff < 0.01

    # ═════════════════════════════════════════════════════════════════════
    # GRAD NORM CHECK (10 steps)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GRAD NORM CHECK (10 steps)")
    print("=" * 60)

    grad_norms = {"baseline": [], "fused": []}
    for mode_name, use_tri in [("baseline", False), ("fused", True)]:
        globals()['_USE_TRITON_ATTN'] = use_tri
        torch.manual_seed(42)
        model = AttnMLPBlock().to(device)

        for step in range(10):
            torch.manual_seed(1000 + step)
            x = torch.randn(Bv, Tv, MODEL_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            target = torch.randn(Bv, Tv, MODEL_DIM, device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
            loss = F.mse_loss(out.float(), target.float())
            loss.backward()
            gn = x.grad.norm().item()
            grad_norms[mode_name].append(gn)
            model.zero_grad()

        del model
        torch.cuda.empty_cache()

    globals()['_USE_TRITON_ATTN'] = True

    print(f"\n{'Step':>5} | {'GradNorm (baseline)':>20} | {'GradNorm (fused)':>17} | {'Match?':>7}")
    print("-" * 60)
    max_grad_pct = 0.0
    all_grad_pass = True
    for i in range(10):
        gb = grad_norms["baseline"][i]
        gf = grad_norms["fused"][i]
        pct = abs(gb - gf) / (gb + 1e-12) * 100
        max_grad_pct = max(max_grad_pct, pct)
        ok = pct < 1.0
        if not ok:
            all_grad_pass = False
        print(f"  {i+1:>3} | {gb:>20.6f} | {gf:>17.6f} | {'YES' if ok else 'NO':>6}")
    print(f"\nMax grad norm deviation: {max_grad_pct:.4f}%")

    # ═════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═════════════════════════════════════════════════════════════════════
    attn_speedup_pass = attn_speedup >= 1.0
    overall = q_pass and k_pass and loss_pass and all_grad_pass and attn_speedup_pass

    print("\n")
    print("+====================================================+")
    print("|  ATTENTION FUSION LOCAL VALIDATION REPORT           |")
    print("+====================================================+")
    print(f"| Q cos_sim:                {cos_q:.6f}    {'[PASS]' if q_pass else '[FAIL]':>8} |")
    print(f"| K cos_sim:                {cos_k:.6f}    {'[PASS]' if k_pass else '[FAIL]':>8} |")
    print(f"| Attn Q+K speedup:         {attn_speedup:.2f}x       {'[PASS]' if attn_speedup_pass else '[FAIL]':>8} |")
    print(f"| Loss diff @ 200 steps:    {max_loss_diff:.6f}    {'[PASS]' if loss_pass else '[FAIL]':>8} |")
    print(f"| Grad norm match:          {max_grad_pct:.2f}%       {'[PASS]' if all_grad_pass else '[FAIL]':>8} |")
    print(f"| Combined save/step:       {total_save:.2f}ms     [INFO]   |")
    print("+====================================================+")
    print(f"| OVERALL: {'READY FOR H100' if overall else 'NOT READY':^39}|")
    print("+====================================================+")
