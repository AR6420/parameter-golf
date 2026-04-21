"""W8A8 MLP_up — correctness + timing validation.

Reference = "quant-aware bf16": apply the exact same per-token A + per-channel W
quantize-then-dequantize pipeline, then run the standard bf16 LeakyReLU^2 math.
This isolates kernel accuracy (quant error is shared between reference and
kernel — we're measuring the kernel's fusion/rounding, not the quant itself).

Uses the STE-identity trick so the grad_w check against the reference is
meaningful (without STE, backward cuts through round()/to(int8) and gives
zero grad — see Phase 2A debug history)."""
import os
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch, torch.nn.functional as F
from w8a8_core import (
    fused_mlp_up_w8a8, quantize_a8_per_token, quantize_w8_per_channel,
    _launch_mlp_up_w8a8,
)

device = 'cuda'
torch.manual_seed(0)

M, K, N = 4096, 512, 1536     # MLP_up production shape


def ste_dequant_w(w_bf16):
    w_i8, s = quantize_w8_per_channel(w_bf16)
    w_deq = (w_i8.float() * s.unsqueeze(-1).float()).to(torch.bfloat16)
    return (w_deq - w_bf16).detach() + w_bf16


def ste_dequant_x(x_bf16):
    # Per-token: reshape to 2D then restore
    orig = x_bf16.shape
    x_i8, s = quantize_a8_per_token(x_bf16)
    x_deq = (x_i8.float() * s.float().unsqueeze(-1)).to(torch.bfloat16).reshape(orig)
    return (x_deq - x_bf16).detach() + x_bf16


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# ── correctness ─────────────────────────────────────────────────────────────

x = torch.randn(2, M // 2, K, device=device, dtype=torch.bfloat16).detach().requires_grad_()
w = torch.randn(N, K, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()

# STE-aware reference: forward uses quant-dequant (captures quant error), backward uses identity
ref = F.leaky_relu(F.linear(ste_dequant_x(x), ste_dequant_w(w)), negative_slope=0.5).square()
out = fused_mlp_up_w8a8(x, w)

cos_fwd = cos(ref, out)
max_abs = (ref - out).abs().max().item()
max_rel = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()

g = torch.randn_like(out)
x.grad = None; w.grad = None
out.backward(g)
gx_out, gw_out = x.grad.clone(), w.grad.clone()
x.grad = None; w.grad = None
ref.backward(g)
cos_gx = cos(x.grad, gx_out)
cos_gw = cos(w.grad, gw_out)

# STE grad_w check: against the pure-bf16 path (no quant) — STE should make them match closely
x.grad = None; w.grad = None
pure_ref = F.leaky_relu(F.linear(x, w), negative_slope=0.5).square()
pure_ref.backward(g)
gw_pure = w.grad.clone()
cos_gw_ste = cos(gw_pure, gw_out)
diff_gw_ste = (gw_pure - gw_out).abs().max().item() / (gw_pure.abs().max().item() + 1e-12)

# ── timing ──────────────────────────────────────────────────────────────────

def bench(fn, warmup=30, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# Raw-kernel timing (pre-quantized — measures only the int8 GEMM + epilogue)
x2d = torch.randn(M, K, device=device, dtype=torch.bfloat16)
w_raw = torch.randn(N, K, device=device, dtype=torch.bfloat16).mul_(0.02)
x_i8, x_s = quantize_a8_per_token(x2d)
w_i8, w_s = quantize_w8_per_channel(w_raw)

ms_bf = bench(lambda: F.leaky_relu(F.linear(x2d, w_raw), negative_slope=0.5).square())
ms_w8a8_raw = bench(lambda: _launch_mlp_up_w8a8(x_i8, w_i8, x_s, w_s))

# Compare to W8A16 (Phase 2A kernel, imported for reference)
try:
    from w8a16_core import _launch_mlp_up as _launch_mlp_up_w8a16, quantize_w8a16
    w_i8_w16, w_s_w16 = quantize_w8a16(w_raw)
    ms_w8a16_raw = bench(lambda: _launch_mlp_up_w8a16(x2d, w_i8_w16, w_s_w16))
except Exception as _e:
    ms_w8a16_raw = float('nan')

# Full autograd-wrapped timing (includes quant-per-call and STE backward setup)
xg = torch.randn(M, K, device=device, dtype=torch.bfloat16).detach().requires_grad_()
wg = torch.randn(N, K, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()
ms_autograd_fwd = bench(lambda: fused_mlp_up_w8a8(xg, wg))

# ── report ──────────────────────────────────────────────────────────────────

print()
print(f"MLP_up W8A8 correctness (M={M}, K={K}, N={N})")
print(f"  cos_fwd:                 {cos_fwd:.6f}   (target > 0.995)    {'PASS' if cos_fwd > 0.995 else 'FAIL'}")
print(f"  cos_grad_x:              {cos_gx:.6f}   (target > 0.99)     {'PASS' if cos_gx > 0.99 else 'FAIL'}")
print(f"  cos_grad_w (vs STE-ref): {cos_gw:.6f}   (target > 0.995)    {'PASS' if cos_gw > 0.995 else 'FAIL'}")
print(f"  cos_grad_w (vs pure bf16, STE identity test): {cos_gw_ste:.6f}")
print(f"  grad_w max_rel_diff_ste: {diff_gw_ste:.3e}  (target < 1e-3)    {'PASS' if diff_gw_ste < 1e-3 else 'FAIL'}")
print(f"  max_abs_diff:            {max_abs:.3e}")
print(f"  max_rel_err:             {max_rel:.3e}   (sensitive to near-zero outputs)")
print()
print(f"MLP_up W8A8 timing")
print(f"  bf16 baseline:           {ms_bf:.4f} ms")
print(f"  W8A16 raw kernel:        {ms_w8a16_raw:.4f} ms   ({ms_bf/ms_w8a16_raw:.2f}x vs bf16)")
print(f"  W8A8 raw kernel:         {ms_w8a8_raw:.4f} ms   ({ms_bf/ms_w8a8_raw:.2f}x vs bf16, {ms_w8a16_raw/ms_w8a8_raw:.2f}x vs W8A16)")
print(f"  W8A8 autograd-wrapped:   {ms_autograd_fwd:.4f} ms (includes per-call quant overhead)")

import json
with open('forgefuse_w8a8_mlpup_validation.json', 'w') as f:
    json.dump({
        'cos_fwd': cos_fwd, 'cos_grad_x': cos_gx, 'cos_grad_w': cos_gw,
        'cos_grad_w_ste_vs_pure_bf16': cos_gw_ste,
        'grad_w_max_rel_diff_ste': diff_gw_ste,
        'max_abs_diff': max_abs, 'max_rel_err': max_rel,
        'ms_bf16': ms_bf, 'ms_w8a16_raw': ms_w8a16_raw,
        'ms_w8a8_raw': ms_w8a8_raw, 'ms_w8a8_autograd': ms_autograd_fwd,
    }, f, indent=2)
