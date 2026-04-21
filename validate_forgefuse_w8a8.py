"""ForgeFuse Phase 2B — 200-step W8A8 vs bf16 comparison.

Same protocol as Phase 2A (W8A16). Tolerances are looser because activation
quantization adds error on top of weight quantization:
  - Step 200 gap < 0.10 absolute
  - Trend stable or decreasing over final 50 steps
  - No NaN, no gradient explosion
  - Grad norm agreement within 1%
"""
import os, sys, types, json, math
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch
import torch.nn.functional as F

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

device = 'cuda'
CFG = dict(
    vocab_size=256, num_layers=4, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
    logit_softcap=30.0, rope_base=1024.0, qk_gain_init=1.0, rope_dims=16,
)
B, T, STEPS, SEED = 2, 128, 200, 1337


def build():
    torch.manual_seed(SEED)
    m = tg.GPT(**CFG).to(device)
    for p in [m.qkv_bank, m.out_bank, m.mlp_up_bank, m.mlp_down_bank]:
        p.data = p.data.float()
    return m


def batch(step):
    g = torch.Generator(device=device).manual_seed(SEED + step)
    ids = torch.randint(0, CFG['vocab_size'], (B, T + 1), generator=g, device=device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def run(mode):
    tg._USE_W8A8_QAT = (mode == 'w8a8')
    m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses, grad_norms = [], []
    any_nan = False
    for step in range(1, STEPS + 1):
        ids, tgt = batch(step)
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = m(ids, tgt)
        loss.backward()
        total = 0.0
        for p in m.parameters():
            if p.grad is not None:
                total += p.grad.float().norm().item() ** 2
        gn = math.sqrt(total)
        if math.isnan(loss.item()) or math.isnan(gn):
            any_nan = True
            break
        opt.step()
        losses.append(loss.item())
        grad_norms.append(gn)
    del m, opt
    torch.cuda.empty_cache()
    return losses, grad_norms, any_nan


print("ForgeFuse Phase 2B — W8A8 vs bf16 (200 steps, same seed, same data)")
print("Model:", CFG)
print()

losses_bf, gn_bf, nan_bf = run('bf16')
losses_w8, gn_w8, nan_w8 = run('w8a8')

checkpoints = [1, 10, 50, 100, 150, 200]
print(f"{'Step':>5} | {'Loss (bf16)':>12} | {'Loss (W8A8)':>12} | {'Gap':>10} | {'|g| bf16':>10} | {'|g| w8a8':>10} | {'|g| diff %':>11}")
print('-' * 90)
for s in checkpoints:
    if s <= len(losses_bf) and s <= len(losses_w8):
        lb, lw = losses_bf[s-1], losses_w8[s-1]
        gb, gw = gn_bf[s-1], gn_w8[s-1]
        gd = abs(gb - gw) / (gb + 1e-12) * 100
        print(f"{s:>5} | {lb:>12.6f} | {lw:>12.6f} | {lw-lb:>+10.6f} | {gb:>10.4f} | {gw:>10.4f} | {gd:>10.3f}%")

gaps = [lw - lb for lb, lw in zip(losses_bf, losses_w8)]
first_50 = gaps[:50]
last_50 = gaps[-50:]
mean_first = sum(first_50) / len(first_50) if first_50 else 0.0
mean_last = sum(last_50) / len(last_50) if last_50 else 0.0
abs_last = sum(abs(g) for g in last_50) / len(last_50) if last_50 else 0.0
max_abs_gap = max(abs(g) for g in gaps) if gaps else 0.0

# Grad norm agreement %
if gn_bf and gn_w8:
    max_gdiff = max(abs(a-b)/(a+1e-12)*100 for a,b in zip(gn_bf, gn_w8))
else:
    max_gdiff = float('inf')

if abs(mean_last) < abs(mean_first) * 1.2 + 1e-3:
    trend = "stable or decreasing"
else:
    trend = "growing"

print()
print(f"Max abs gap (all 200 steps): {max_abs_gap:.6f}")
print(f"Mean gap over first 50 steps: {mean_first:+.6f}")
print(f"Mean gap over last 50 steps:  {mean_last:+.6f}  (abs: {abs_last:.6f})")
print(f"Trend: {trend}")
print(f"Max grad-norm %-diff:         {max_gdiff:.3f}%  (target < 1%)")
print(f"NaN? bf16={nan_bf}, w8a8={nan_w8}")

final_gap = gaps[-1] if gaps else float('inf')
passed = (not nan_w8 and not nan_bf and abs(final_gap) < 0.10 and
          "stable" in trend and max_gdiff < 1.0)
print(f"Final gap (step {len(gaps)}): {final_gap:+.6f}")
print()
print(f"RESULT: {'PASS' if passed else 'FAIL'}")

with open('forgefuse_w8a8_validation.json', 'w') as f:
    json.dump({
        'losses_bf16': losses_bf, 'losses_w8a8': losses_w8,
        'grad_norms_bf16': gn_bf, 'grad_norms_w8a8': gn_w8,
        'max_abs_gap': max_abs_gap,
        'mean_gap_first_50': mean_first, 'mean_gap_last_50': mean_last,
        'final_gap': final_gap, 'trend': trend,
        'max_grad_norm_pct_diff': max_gdiff,
        'nan_bf16': nan_bf, 'nan_w8a8': nan_w8, 'passed': passed,
    }, f, indent=2)
