"""W8A8 all-4-kernels correctness + timing.

Reference for each kernel = STE-aware "quant-then-dequant" forward combined
with the corresponding bf16 epilogue. Same pattern as W8A16 test."""
import os, json
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch, torch.nn.functional as F
from w8a8_core import (
    fused_qkv_w8a8, fused_out_proj_w8a8, fused_mlp_up_w8a8, fused_mlp_down_w8a8,
    quantize_a8_per_token, quantize_w8_per_channel,
    _launch_qkv_w8a8, _launch_out_proj_w8a8, _launch_mlp_up_w8a8, _launch_mlp_down_w8a8,
)

device = 'cuda'
torch.manual_seed(0)

B, T = 2, 2048
DIM, KV_DIM, MLP_DIM = 512, 256, 1536


def ste_w(w_bf16):
    w_i8, s = quantize_w8_per_channel(w_bf16)
    w_deq = (w_i8.float() * s.unsqueeze(-1).float()).to(torch.bfloat16)
    return (w_deq - w_bf16).detach() + w_bf16


def ste_a(x_bf16):
    orig = x_bf16.shape
    x_i8, s = quantize_a8_per_token(x_bf16)
    x_deq = (x_i8.float() * s.float().unsqueeze(-1)).to(torch.bfloat16).reshape(orig)
    return (x_deq - x_bf16).detach() + x_bf16


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def bench(fn, warmup=30, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


results = []

# ============================================================================
# Kernel 1: QKV
# ============================================================================
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()
w = torch.randn(DIM + 2*KV_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()

ref = F.linear(ste_a(x), ste_w(w))
out = fused_qkv_w8a8(x, w)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
x.grad = None; w.grad = None; out.backward(g)
gx_o, gw_o = x.grad.clone(), w.grad.clone()
x.grad = None; w.grad = None; ref.backward(g)
cos_gx = cos(x.grad, gx_o)
cos_gw = cos(w.grad, gw_o)

# STE identity test: W8A8 grad_w should equal pure-bf16 grad_w
x.grad = None; w.grad = None
F.linear(x, w).sum().backward()  # trigger graph build
x.grad = None; w.grad = None
pure = F.linear(x, w)
pure.backward(g)
ste_diff = (w.grad - gw_o).abs().max().item() / (w.grad.abs().max().item() + 1e-12)

ms_bf = bench(lambda: F.linear(x, w))
x2d = x.reshape(-1, DIM).detach().contiguous()
x_i8, x_s = quantize_a8_per_token(x2d)
w_i8, w_s = quantize_w8_per_channel(w.detach())
ms_w8 = bench(lambda: _launch_qkv_w8a8(x_i8, w_i8, x_s, w_s))
results.append(('QKV', (4096, DIM, DIM+2*KV_DIM), cos_fwd, cos_gx, cos_gw, ste_diff, ms_bf, ms_w8))

# ============================================================================
# Kernel 2: OutProj
# ============================================================================
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()
w = torch.randn(DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()
attn_scale = (torch.randn(DIM, device=device, dtype=torch.bfloat16) * 0.5 + 1.0).detach().requires_grad_()
residual = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()

ref = residual + attn_scale * F.linear(ste_a(x), ste_w(w))
out = fused_out_proj_w8a8(x, w, attn_scale, residual)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
for t in [x, w, attn_scale, residual]: t.grad = None
out.backward(g)
grads_o = [x.grad.clone(), w.grad.clone(), attn_scale.grad.clone(), residual.grad.clone()]
for t in [x, w, attn_scale, residual]: t.grad = None
ref.backward(g)
cos_gx = cos(x.grad, grads_o[0])
cos_gw = cos(w.grad, grads_o[1])
cos_gs = cos(attn_scale.grad, grads_o[2])
cos_gr = cos(residual.grad, grads_o[3])
cos_gw_worst = min(cos_gx, cos_gw, cos_gs, cos_gr)

# STE identity: pure bf16 grad_w
for t in [x, w, attn_scale, residual]: t.grad = None
(residual + attn_scale * F.linear(x, w)).backward(g)
ste_diff = (w.grad - grads_o[1]).abs().max().item() / (w.grad.abs().max().item() + 1e-12)

ms_bf = bench(lambda: residual + attn_scale * F.linear(x, w))
x2d = x.reshape(-1, DIM).detach().contiguous()
r2d = residual.reshape(-1, DIM).detach().contiguous()
x_i8, x_s = quantize_a8_per_token(x2d)
w_i8, w_s = quantize_w8_per_channel(w.detach())
ms_w8 = bench(lambda: _launch_out_proj_w8a8(x_i8, w_i8, x_s, w_s, attn_scale.detach(), r2d))
results.append(('OutProj', (4096, DIM, DIM), cos_fwd, cos_gx, cos_gw_worst, ste_diff, ms_bf, ms_w8))

# ============================================================================
# Kernel 3: MLP_up (already tested, repeat for table completeness)
# ============================================================================
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()
w = torch.randn(MLP_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()

ref = F.leaky_relu(F.linear(ste_a(x), ste_w(w)), negative_slope=0.5).square()
out = fused_mlp_up_w8a8(x, w)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
x.grad = None; w.grad = None; out.backward(g)
gx_o, gw_o = x.grad.clone(), w.grad.clone()
x.grad = None; w.grad = None; ref.backward(g)
cos_gx = cos(x.grad, gx_o)
cos_gw = cos(w.grad, gw_o)

x.grad = None; w.grad = None
F.leaky_relu(F.linear(x, w), negative_slope=0.5).square().backward(g)
ste_diff = (w.grad - gw_o).abs().max().item() / (w.grad.abs().max().item() + 1e-12)

ms_bf = bench(lambda: F.leaky_relu(F.linear(x, w), negative_slope=0.5).square())
x2d = x.reshape(-1, DIM).detach().contiguous()
x_i8, x_s = quantize_a8_per_token(x2d)
w_i8, w_s = quantize_w8_per_channel(w.detach())
ms_w8 = bench(lambda: _launch_mlp_up_w8a8(x_i8, w_i8, x_s, w_s))
results.append(('MLP_up', (4096, DIM, MLP_DIM), cos_fwd, cos_gx, cos_gw, ste_diff, ms_bf, ms_w8))

# ============================================================================
# Kernel 4: MLP_down
# ============================================================================
x = torch.randn(B, T, MLP_DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()
w = torch.randn(DIM, MLP_DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()
mlp_scale = (torch.randn(DIM, device=device, dtype=torch.bfloat16) * 0.5 + 1.0).detach().requires_grad_()
residual = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).detach().requires_grad_()

ref = residual + mlp_scale * F.linear(ste_a(x), ste_w(w))
out = fused_mlp_down_w8a8(x, w, mlp_scale, residual)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
for t in [x, w, mlp_scale, residual]: t.grad = None
out.backward(g)
grads_o = [x.grad.clone(), w.grad.clone(), mlp_scale.grad.clone(), residual.grad.clone()]
for t in [x, w, mlp_scale, residual]: t.grad = None
ref.backward(g)
cos_gx = cos(x.grad, grads_o[0])
cos_gw = cos(w.grad, grads_o[1])
cos_gs = cos(mlp_scale.grad, grads_o[2])
cos_gr = cos(residual.grad, grads_o[3])
cos_gw_worst = min(cos_gx, cos_gw, cos_gs, cos_gr)

for t in [x, w, mlp_scale, residual]: t.grad = None
(residual + mlp_scale * F.linear(x, w)).backward(g)
ste_diff = (w.grad - grads_o[1]).abs().max().item() / (w.grad.abs().max().item() + 1e-12)

ms_bf = bench(lambda: residual + mlp_scale * F.linear(x, w))
x2d = x.reshape(-1, MLP_DIM).detach().contiguous()
r2d = residual.reshape(-1, DIM).detach().contiguous()
x_i8, x_s = quantize_a8_per_token(x2d)
w_i8, w_s = quantize_w8_per_channel(w.detach())
ms_w8 = bench(lambda: _launch_mlp_down_w8a8(x_i8, w_i8, x_s, w_s, mlp_scale.detach(), r2d))
results.append(('MLP_down', (4096, MLP_DIM, DIM), cos_fwd, cos_gx, cos_gw_worst, ste_diff, ms_bf, ms_w8))

# ============================================================================
# Report
# ============================================================================
print()
print(f"{'kernel':>9} | {'(M,K,N)':>19} | {'cos_fwd':>9} | {'cos_gx':>8} | {'cos_worst':>9} | {'STE_diff':>9} | {'ms_bf16':>7} | {'ms_w8a8':>7} | {'ratio':>6} | {'status':>6}")
print('-' * 128)
pass_all = True
for (name, shape, c_fwd, c_gx, c_worst, ste, ms_bf, ms_w) in results:
    fwd_ok = c_fwd > 0.995
    gx_ok = c_gx > 0.99
    gw_ok = c_worst > 0.995
    ste_ok = ste < 1e-3
    status = 'PASS' if (fwd_ok and gx_ok and gw_ok and ste_ok) else 'FAIL'
    if status == 'FAIL': pass_all = False
    print(f"{name:>9} | {str(shape):>19} | {c_fwd:>9.6f} | {c_gx:>8.6f} | {c_worst:>9.6f} | "
          f"{ste:>9.2e} | {ms_bf:>6.4f} | {ms_w:>6.4f} | {ms_bf/ms_w:>5.2f}x | {status:>6}")

print()
print(f"Overall: {'PASS' if pass_all else 'FAIL'}")

with open('forgefuse_w8a8_kernels_validation.json', 'w') as f:
    json.dump([{
        'name': n, 'shape': s, 'cos_fwd': c_fwd, 'cos_grad_x': c_gx, 'cos_worst': c_worst,
        'ste_diff': ste, 'ms_bf16': ms_bf, 'ms_w8a8': ms_w, 'ratio': ms_bf/ms_w,
    } for (n, s, c_fwd, c_gx, c_worst, ste, ms_bf, ms_w) in results], f, indent=2)
