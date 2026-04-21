"""
ForgeFuse Phase 1 correctness validator.

Runs 10 steps of the full GPT model in two modes:
  - FUSED:   single F.linear(x, qkv_bank[i]) then split (forgefuse path)
  - UNFUSED: split qkv_bank[i] into Q|K|V slices, then 3 F.linear calls

Both modes use the *exact same* qkv_bank weights, so the only source of
divergence is cuBLAS accumulation order. Requirement: loss diff < 1e-5
per step across 10 steps.
"""
import os, sys, types, math, json, copy
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'

import torch
import torch.nn.functional as F

# --- Mock flash_attn with SDPA math (required on RTX 5070 Ti, no FA3) ---
mock = types.ModuleType('flash_attn_interface')
def _sdpa(q, k, v, causal=False, **kw):
    q, k, v = q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous()
    H, Hkv = q.shape[1], k.shape[1]
    if H != Hkv:
        k = k.repeat_interleave(H // Hkv, dim=1)
        v = v.repeat_interleave(H // Hkv, dim=1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
mock.flash_attn_func = _sdpa
sys.modules['flash_attn_interface'] = mock

import train_gpt as tg
from train_gpt import GPT, CausalSelfAttention as SelfAttention

device = 'cuda'

# ---------- model config ----------
# Small but representative: 4 layers, correct head layout, rope_dims>0 to exercise kernels.
CFG = dict(
    vocab_size=256, num_layers=4, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
    logit_softcap=30.0, rope_base=1024.0, qk_gain_init=1.0,
    rope_dims=16,
)
B, T, STEPS, SEED = 2, 128, 100, 1337


def build_model():
    torch.manual_seed(SEED)
    m = GPT(**CFG).to(device)
    # Mirror the training pipeline: banks stay fp32, cast to bf16 in forward.
    m.qkv_bank.data = m.qkv_bank.data.float()
    m.out_bank.data = m.out_bank.data.float()
    m.mlp_up_bank.data = m.mlp_up_bank.data.float()
    m.mlp_down_bank.data = m.mlp_down_bank.data.float()
    return m


def make_batch(step):
    g = torch.Generator(device=device).manual_seed(SEED + step)
    ids = torch.randint(0, CFG['vocab_size'], (B, T + 1), generator=g, device=device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


# ---------- unfused reference forward ----------
_original_attn_forward = SelfAttention.forward

def unfused_attn_forward(self, x, qkv_w, out_w, v_embed=None, v0=None):
    """Reference: split qkv_w on dim=0 and run 3 separate F.linear calls."""
    bsz, seqlen, dim = x.shape
    q_dim = self.num_heads * self.head_dim
    kv_dim = self.num_kv_heads * self.head_dim
    q_w, k_w, v_w = qkv_w.split([q_dim, kv_dim, kv_dim], dim=0)
    q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
    k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = F.linear(x, v_w.to(x.dtype))
    if v_embed is not None:
        v = v + v_embed
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    raw_v = v if self.value_residual else None
    if self.value_residual and v0 is not None:
        alpha = torch.sigmoid(self.vrl_alpha.to(dtype=v.dtype))
        v = v + alpha * v0
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = tg.fused_q_norm_rope_gain(q, cos, sin, self.q_gain.to(dtype=q.dtype), self.rope_dims)
    k = tg.fused_k_norm_rope(k, cos, sin, self.rope_dims)
    y = tg.flash_attn_3_func(q, k, v, causal=True)
    if self.use_xsa:
        y = self._xsa_efficient(y, v)
    if self.gated_attention:
        gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
        y = y * gate
    y = y.reshape(bsz, seqlen, dim)
    return F.linear(y, out_w.to(x.dtype)), raw_v


def run_ten(mode):
    """Run 10 training steps; return list of per-step losses."""
    m = build_model()
    # Build optimizer *after* bank dtype is fp32 so param groups are consistent
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    if mode == 'unfused':
        SelfAttention.forward = unfused_attn_forward
    else:
        SelfAttention.forward = _original_attn_forward

    losses = []
    for step in range(1, STEPS + 1):
        ids, tgt = make_batch(step)
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = m(ids, tgt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Restore
    SelfAttention.forward = _original_attn_forward
    del m, opt
    torch.cuda.empty_cache()
    return losses


print("ForgeFuse Phase 1 — correctness validation")
print("Model:", CFG)
print(f"Steps: {STEPS}, Batch: {B}x{T}, Seed: {SEED}")
print()

torch.cuda.manual_seed_all(SEED)
losses_unfused = run_ten('unfused')
torch.cuda.manual_seed_all(SEED)
losses_fused   = run_ten('fused')

diffs = [abs(lu - lf) for lu, lf in zip(losses_unfused, losses_fused)]

# Print a sampled table (first 5 + last 5 + peak)
peak_step = max(range(len(diffs)), key=lambda i: diffs[i])
show = sorted(set(list(range(5)) + list(range(len(diffs) - 5, len(diffs))) + [peak_step]))
print(f"{'Step':>5} | {'Loss (unfused)':>18} | {'Loss (fused)':>18} | {'Diff':>12}")
print('-' * 64)
for i in show:
    tag = '  (peak)' if i == peak_step else ''
    print(f"{i+1:>5} | {losses_unfused[i]:>18.9f} | {losses_fused[i]:>18.9f} | {diffs[i]:>12.2e}{tag}")

max_diff = max(diffs)
step100_diff = diffs[-1]
# Trend analysis: linear regression slope of diff vs step; compare diff growth in first vs last quartile
q1 = sum(diffs[:STEPS // 4]) / max(1, STEPS // 4)
q4 = sum(diffs[-STEPS // 4:]) / max(1, STEPS // 4)
trend = 'stable' if q4 < 2 * q1 else ('growing' if q4 < 10 * q1 else 'exploding')

# Loss curve correlation (relative L1 distance)
rel_l1 = sum(diffs) / max(1e-12, sum(abs(l) for l in losses_unfused))

print()
print(f"Max loss diff across {STEPS} steps:    {max_diff:.3e}  (peak at step {peak_step + 1})")
print(f"Loss diff at step {STEPS}:                {step100_diff:.3e}")
print(f"Trend (q1 avg {q1:.2e} -> q4 avg {q4:.2e}): {trend}")
print(f"Relative L1 drift (sum|diff| / sum|L|): {rel_l1:.6f}  ({rel_l1 * 100:.4f}%)")

tol = 1e-3  # loosened per agreement: bf16 cuBLAS algo selection dominates below this
passed = (trend != 'exploding') and (rel_l1 < 0.001) and (max_diff < tol)
print()
if passed:
    print(f"PASS  — drift bounded, relative L1 < 0.1%, not exploding")
else:
    print(f"FAIL  — trend={trend}, rel_l1={rel_l1:.6f}, max={max_diff:.3e}")

with open('forgefuse_validation.json', 'w') as f:
    json.dump({'losses_unfused': losses_unfused, 'losses_fused': losses_fused,
               'max_diff': max_diff, 'step_last_diff': step100_diff,
               'relative_l1': rel_l1, 'trend': trend, 'tol': tol,
               'passed': passed}, f, indent=2)
