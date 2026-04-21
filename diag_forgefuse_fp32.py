"""Diagnostic: run the 10-step validator in pure fp32 (no bf16 autocast).
If the math is algebraically identical, fp32 should give bit-exact match.
A diff > 0 in fp32 would indicate a genuine routing/init bug."""
import os, sys, types
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch, torch.nn.functional as F
mock = types.ModuleType('flash_attn_interface')
def _sdpa(q, k, v, causal=False, **kw):
    q, k, v = q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous()
    H, Hkv = q.shape[1], k.shape[1]
    if H != Hkv:
        k = k.repeat_interleave(H // Hkv, dim=1); v = v.repeat_interleave(H // Hkv, dim=1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
mock.flash_attn_func = _sdpa
sys.modules['flash_attn_interface'] = mock

# Disable TF32 so matmul is deterministic in fp32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import train_gpt as tg
from train_gpt import GPT, CausalSelfAttention as SelfAttention

device = 'cuda'
CFG = dict(vocab_size=256, num_layers=4, model_dim=512, num_heads=8, num_kv_heads=4,
           mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
           logit_softcap=30.0, rope_base=1024.0, qk_gain_init=1.0, rope_dims=16)
B, T, STEPS, SEED = 2, 128, 10, 1337

def build():
    torch.manual_seed(SEED)
    return GPT(**CFG).to(device).float()

def batch(step):
    g = torch.Generator(device=device).manual_seed(SEED + step)
    ids = torch.randint(0, CFG['vocab_size'], (B, T+1), generator=g, device=device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()

_orig = SelfAttention.forward
def unfused(self, x, qkv_w, out_w, v_embed=None, v0=None):
    bsz, seqlen, dim = x.shape
    qd = self.num_heads * self.head_dim; kd = self.num_kv_heads * self.head_dim
    qw, kw, vw = qkv_w.split([qd, kd, kd], dim=0)
    q = F.linear(x, qw.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
    k = F.linear(x, kw.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = F.linear(x, vw.to(x.dtype))
    if v_embed is not None: v = v + v_embed
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    raw_v = v if self.value_residual else None
    if self.value_residual and v0 is not None:
        alpha = torch.sigmoid(self.vrl_alpha.to(dtype=v.dtype)); v = v + alpha * v0
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = tg.fused_q_norm_rope_gain(q, cos, sin, self.q_gain.to(dtype=q.dtype), self.rope_dims)
    k = tg.fused_k_norm_rope(k, cos, sin, self.rope_dims)
    y = tg.flash_attn_3_func(q, k, v, causal=True)
    if self.use_xsa: y = self._xsa_efficient(y, v)
    if self.gated_attention:
        gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1); y = y * gate
    y = y.reshape(bsz, seqlen, dim)
    return F.linear(y, out_w.to(x.dtype)), raw_v

def run(mode):
    m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    SelfAttention.forward = unfused if mode == 'unfused' else _orig
    losses = []
    for s in range(1, STEPS+1):
        ids, tgt = batch(s)
        opt.zero_grad()
        loss = m(ids, tgt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    SelfAttention.forward = _orig
    del m, opt; torch.cuda.empty_cache()
    return losses

print("FP32 diagnostic (no autocast, TF32 off)")
lu = run('unfused')
lf = run('fused')
print(f"{'Step':>4} | {'unfused':>20} | {'fused':>20} | {'diff':>12}")
for i, (u, f_) in enumerate(zip(lu, lf), 1):
    print(f"{i:>4} | {u:>20.12f} | {f_:>20.12f} | {abs(u-f_):>12.2e}")
print(f"\nMax diff (fp32): {max(abs(u-f_) for u,f_ in zip(lu,lf)):.3e}")
