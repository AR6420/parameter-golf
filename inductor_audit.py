"""
Phase 0: Inductor Fusion Audit
Builds a minimal model matching SOTA architecture and inspects what torch.compile/Inductor
fuses automatically. Replaces FlashAttn3 with SDPA for local compatibility.
"""
import os, math, torch, torch.nn.functional as F
from torch import Tensor, nn

# ── Architecture params matching SOTA (PR #1019) ──
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = MODEL_DIM // NUM_HEADS  # 64
KV_DIM = NUM_KV_HEADS * HEAD_DIM   # 256
MLP_DIM = int(MODEL_DIM * 3.0)     # 1536
NUM_LAYERS = 11
SEQ_LEN = 1024
BATCH = 4  # sequences (keep small for 12GB VRAM audit)

# ── Minimal model components ──

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    """Partial RoPE matching SOTA implementation."""
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, rope_dims=16):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        # Shape: [1, seq_len, 1, rope_dims//2] -- broadcast for [B, T, H, D]
        cos = freqs.cos()[None, :, None, :].to(dtype)
        sin = freqs.sin()[None, :, None, :].to(dtype)
        return cos, sin

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.q_gain = nn.Parameter(torch.ones(NUM_HEADS))
        self.rope_dims = 16  # partial RoPE: 16 of 64 dims
        self.rotary = Rotary(HEAD_DIM, rope_dims=self.rope_dims)

    def forward(self, x, q_w, k_w, v_w, out_w):
        bsz, seqlen, dim = x.shape
        # 3 separate GEMMs for Q, K, V
        q = F.linear(x, q_w).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        # Element-wise: RMSNorm on Q, K
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Element-wise: RoPE
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        # Element-wise: Q gain
        q = q * self.q_gain[None, None, :, None]
        # Attention (SDPA instead of FlashAttn3)
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # GQA: repeat K, V
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, seqlen, dim)
        # Output GEMM
        return F.linear(y, out_w)

class MLP(nn.Module):
    def forward(self, x, up_w, down_w):
        # GEMM 1 + element-wise activation
        x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5)
        # Element-wise square + GEMM 2
        return F.linear(x.square(), down_w)

class Block(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()
        self.attn_scale = nn.Parameter(torch.ones(MODEL_DIM))
        self.mlp_scale = nn.Parameter(torch.ones(MODEL_DIM))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(MODEL_DIM), torch.zeros(MODEL_DIM))))
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # Attn path: RMSNorm -> scale -> GEMM(Q,K,V) -> ... -> GEMM(Out) -> residual add
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w)
        x_out = x_in + self.attn_scale[None, None, :] * attn_out
        # MLP path: RMSNorm -> scale -> GEMM(Up) -> LeakyReLU^2 -> GEMM(Down) -> residual add
        x_out = x_out + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

class MinimalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(i) for i in range(NUM_LAYERS)])
        self.tok_emb = nn.Embedding(1024, MODEL_DIM)
        self.final_norm = RMSNorm()
        # Parameter banks (matching SOTA)
        self.qo_bank = nn.Parameter(torch.randn(2*NUM_LAYERS, MODEL_DIM, MODEL_DIM) * 0.02)
        self.kv_bank = nn.Parameter(torch.randn(2*NUM_LAYERS, KV_DIM, MODEL_DIM) * 0.02)
        self.mlp_up = nn.Parameter(torch.randn(NUM_LAYERS, MLP_DIM, MODEL_DIM) * 0.02)
        self.mlp_down = nn.Parameter(torch.randn(NUM_LAYERS, MODEL_DIM, MLP_DIM) * 0.02)

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        n = NUM_LAYERS
        for i, blk in enumerate(self.blocks):
            x = blk(x, x0,
                     self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i],
                     self.qo_bank[n+i], self.mlp_up[i], self.mlp_down[i])
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        return F.cross_entropy(logits.view(-1, 1024), input_ids.view(-1))


if __name__ == "__main__":
    import logging
    device = "cuda"
    torch.manual_seed(1337)

    B_T = BATCH * SEQ_LEN  # total tokens

    print("=" * 80)
    print("PHASE 0: INDUCTOR FUSION AUDIT")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Batch tokens: {B_T}, model_dim={MODEL_DIM}, mlp_dim={MLP_DIM}")
    print("=" * 80)

    x_test = torch.randn(B_T, MODEL_DIM, device=device, dtype=torch.bfloat16)
    up_w = torch.randn(MLP_DIM, MODEL_DIM, device=device, dtype=torch.bfloat16)
    down_w = torch.randn(MODEL_DIM, MLP_DIM, device=device, dtype=torch.bfloat16)
    w_512 = torch.randn(MODEL_DIM, MODEL_DIM, device=device, dtype=torch.bfloat16)
    scale_test = torch.randn(MODEL_DIM, device=device, dtype=torch.bfloat16)
    res_test = torch.randn(B_T, MODEL_DIM, device=device, dtype=torch.bfloat16)

    # ── Test 1: MLP (GEMM -> LeakyReLU -> square -> GEMM) ──
    # Q: Does Inductor fuse LeakyReLU+square as GEMM1 epilogue? As GEMM2 prologue?
    print("\n>>> TEST 1: MLP -- GEMM + LeakyReLU^2 + GEMM")
    def mlp_forward(x, up_w, down_w):
        x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5)
        return F.linear(x.square(), down_w)
    torch._dynamo.reset()
    compiled_mlp = torch.compile(mlp_forward, dynamic=False)
    with torch.no_grad():
        out = compiled_mlp(x_test, up_w, down_w)
        print(f"  Output shape: {out.shape}")

    # ── Test 2: RMSNorm -> GEMM (prologue fusion?) ──
    print("\n>>> TEST 2: RMSNorm + GEMM (prologue fusion check)")
    def norm_gemm(x, w):
        return F.linear(F.rms_norm(x, (x.size(-1),)), w)
    torch._dynamo.reset()
    compiled_ng = torch.compile(norm_gemm, dynamic=False)
    with torch.no_grad():
        out = compiled_ng(x_test, up_w)
        print(f"  Output shape: {out.shape}")

    # ── Test 3: GEMM -> scale + residual (epilogue fusion?) ──
    print("\n>>> TEST 3: GEMM + scale*out + residual (epilogue fusion check)")
    def gemm_residual(x, w, scale, residual):
        return residual + scale[None, :] * F.linear(x, w)
    torch._dynamo.reset()
    compiled_gr = torch.compile(gemm_residual, dynamic=False)
    with torch.no_grad():
        out = compiled_gr(x_test, w_512, scale_test, res_test)
        print(f"  Output shape: {out.shape}")

    # ── Test 4: Full MLP path with norm + residual (the complete fusion target) ──
    # RMSNorm(x) -> scale -> GEMM_up -> LeakyReLU -> square -> GEMM_down -> scale·out + residual
    print("\n>>> TEST 4: Full MLP path (RMSNorm -> GEMM -> act^2 -> GEMM -> residual)")
    def full_mlp_path(x, up_w, down_w, mlp_scale, ln_scale_factor):
        normed = F.rms_norm(x, (x.size(-1),)) * ln_scale_factor
        h = F.leaky_relu(F.linear(normed, up_w), negative_slope=0.5)
        out = F.linear(h.square(), down_w)
        return x + mlp_scale[None, :] * out
    torch._dynamo.reset()
    compiled_full_mlp = torch.compile(full_mlp_path, dynamic=False)
    ln_sf = torch.tensor(1.0 / math.sqrt(6), device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = compiled_full_mlp(x_test, up_w, down_w, scale_test, ln_sf)
        print(f"  Output shape: {out.shape}")

    # ── Test 5: Same as Test 4 but with max-autotune ──
    print("\n>>> TEST 5: Full MLP path with mode='max-autotune'")
    torch._dynamo.reset()
    compiled_full_mlp_at = torch.compile(full_mlp_path, mode='max-autotune', dynamic=False)
    with torch.no_grad():
        out = compiled_full_mlp_at(x_test, up_w, down_w, scale_test, ln_sf)
        print(f"  Output shape: {out.shape}")

    # ── Test 6: RMSNorm -> 3x GEMM (QKV fusion check) ──
    print("\n>>> TEST 6: RMSNorm -> Q,K,V GEMMs (multi-GEMM prologue sharing)")
    kv_w = torch.randn(KV_DIM, MODEL_DIM, device=device, dtype=torch.bfloat16)
    def qkv_path(x, q_w, k_w, v_w):
        normed = F.rms_norm(x, (x.size(-1),))
        q = F.linear(normed, q_w)
        k = F.linear(normed, k_w)
        v = F.linear(normed, v_w)
        return q, k, v
    torch._dynamo.reset()
    compiled_qkv = torch.compile(qkv_path, dynamic=False)
    with torch.no_grad():
        q, k, v = compiled_qkv(x_test, w_512, kv_w, kv_w)
        print(f"  Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # ── Test 7: Timing comparison ──
    print("\n>>> TEST 7: Timing -- unfused vs compiled MLP path")
    import time

    def unfused_mlp(x, up_w, down_w, mlp_scale, ln_sf):
        normed = F.rms_norm(x, (x.size(-1),)) * ln_sf
        h = F.leaky_relu(F.linear(normed, up_w), negative_slope=0.5)
        out = F.linear(h.square(), down_w)
        return x + mlp_scale[None, :] * out

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = unfused_mlp(x_test, up_w, down_w, scale_test, ln_sf)
            _ = compiled_full_mlp(x_test, up_w, down_w, scale_test, ln_sf)
    torch.cuda.synchronize()

    # Time unfused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = unfused_mlp(x_test, up_w, down_w, scale_test, ln_sf)
    torch.cuda.synchronize()
    unfused_ms = (time.perf_counter() - t0) / 100 * 1000

    # Time compiled (default)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = compiled_full_mlp(x_test, up_w, down_w, scale_test, ln_sf)
    torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - t0) / 100 * 1000

    # Time compiled (max-autotune)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = compiled_full_mlp_at(x_test, up_w, down_w, scale_test, ln_sf)
    torch.cuda.synchronize()
    autotune_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"  Unfused (eager):     {unfused_ms:.3f} ms")
    print(f"  Compiled (default):  {compiled_ms:.3f} ms")
    print(f"  Compiled (autotune): {autotune_ms:.3f} ms")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE -- analyze inductor_fusion_audit.log for kernel fusion details")
    print("Look for: mm/addmm calls (cuBLAS) vs triton kernels (fused element-wise)")
    print("=" * 80)
