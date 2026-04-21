"""Per-kernel correctness + timing for W8A16 variants.

Reference for each kernel is "quantize weight, dequantize back to bf16, then
run the original bf16 compute". This isolates the kernel's numerical fidelity
(the quantize step is shared between reference and kernel, so any loss from
quant is NOT counted against the kernel — only the kernel's rounding/fusion
inaccuracy is)."""
import os
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch, torch.nn.functional as F
from w8a16_core import (
    fused_qkv_w8a16, fused_out_proj_w8a16, fused_mlp_up_w8a16, fused_mlp_down_w8a16,
    quantize_w8a16,
    _launch_qkv, _launch_out_proj, _launch_mlp_up, _launch_mlp_down,
)

device = 'cuda'
torch.manual_seed(0)

# Production shapes, all with M = B*T = 4096 and model_dim = 512
B, T = 2, 2048         # B*T = 4096
DIM = 512
KV_DIM = 256
MLP_DIM = 1536

# ============================================================================
# Helpers
# ============================================================================

def dequantize(w_bf16):
    """Apply the same quant path the kernel uses, then dequantize to bf16.
    Reference operations use this weight so kernel accuracy is measured in
    isolation from quantization error."""
    w_i8, s = quantize_w8a16(w_bf16)
    return (w_i8.float() * s.unsqueeze(-1).float()).to(torch.bfloat16)


def ste_weight(w_bf16):
    """STE-aware weight for the reference: forward uses dequant(w), backward
    sees identity (d/dw = 1). This lets us compare grads fairly against the
    autograd.Function wrappers which use the same STE."""
    with torch.no_grad():
        w_deq = dequantize(w_bf16)
    # (w_deq - w).detach() + w: value == w_deq, grad w.r.t. w == identity
    return (w_deq - w_bf16).detach() + w_bf16


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


results = []

# ============================================================================
# Kernel 1: QKV
# ============================================================================
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).requires_grad_()
w = torch.randn(DIM + 2 * KV_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()

ref = F.linear(x, ste_weight(w))
out = fused_qkv_w8a16(x, w)
cos_fwd = cos(ref, out)

# Backward: sanity check with a scalar loss
g = torch.randn_like(out)
x.grad = None; w.grad = None
out.backward(g)
out_gx, out_gw = x.grad.clone(), w.grad.clone()
x.grad = None; w.grad = None
ref.backward(g)
cos_gx = cos(x.grad, out_gx)
cos_gw = cos(w.grad, out_gw)

# Timing vs bf16 baseline
ms_bf = bench(lambda: F.linear(x, w))
ms_w  = bench(lambda: fused_qkv_w8a16(x, w))
results.append(('QKV', (4096, DIM, DIM + 2*KV_DIM), cos_fwd, cos_gx, cos_gw, ms_bf, ms_w))

# ============================================================================
# Kernel 2: OutProj
# ============================================================================
# Attn output y has shape [B, T, DIM]. OutProj: [DIM, DIM] weight, attn_scale [DIM].
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).requires_grad_()
w = torch.randn(DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()
attn_scale = (torch.randn(DIM, device=device, dtype=torch.bfloat16) * 0.5 + 1.0).detach().requires_grad_()
residual = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).requires_grad_()

ref = residual + attn_scale * F.linear(x, ste_weight(w))
out = fused_out_proj_w8a16(x, w, attn_scale, residual)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
for t in [x, w, attn_scale, residual]: t.grad = None
out.backward(g)
out_grads = [x.grad.clone(), w.grad.clone(), attn_scale.grad.clone(), residual.grad.clone()]
for t in [x, w, attn_scale, residual]: t.grad = None
ref.backward(g)
cos_gx = cos(x.grad, out_grads[0])
cos_gw = cos(w.grad, out_grads[1])
cos_gs = cos(attn_scale.grad, out_grads[2])
cos_gr = cos(residual.grad, out_grads[3])
cos_gw = min(cos_gx, cos_gw, cos_gs, cos_gr)  # report worst

ms_bf = bench(lambda: residual + attn_scale * F.linear(x, w))
ms_w  = bench(lambda: fused_out_proj_w8a16(x, w, attn_scale, residual))
results.append(('OutProj', (4096, DIM, DIM), cos_fwd, cos_gx, cos_gw, ms_bf, ms_w))

# ============================================================================
# Kernel 3: MLP_up (LeakyReLU(0.5)^2)
# ============================================================================
x = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).requires_grad_()
w = torch.randn(MLP_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()

ref = F.leaky_relu(F.linear(x, ste_weight(w)), negative_slope=0.5).square()
out = fused_mlp_up_w8a16(x, w)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
x.grad = None; w.grad = None
out.backward(g)
out_gx, out_gw = x.grad.clone(), w.grad.clone()
x.grad = None; w.grad = None
ref.backward(g)
cos_gx = cos(x.grad, out_gx)
cos_gw = cos(w.grad, out_gw)

ms_bf = bench(lambda: F.leaky_relu(F.linear(x, w), negative_slope=0.5).square())
ms_w  = bench(lambda: fused_mlp_up_w8a16(x, w))
results.append(('MLP_up', (4096, DIM, MLP_DIM), cos_fwd, cos_gx, cos_gw, ms_bf, ms_w))

# ============================================================================
# Kernel 4: MLP_down (residual + mlp_scale * gemm)
# ============================================================================
x = torch.randn(B, T, MLP_DIM, device=device, dtype=torch.bfloat16).requires_grad_()
w = torch.randn(DIM, MLP_DIM, device=device, dtype=torch.bfloat16).mul_(0.02).requires_grad_()
mlp_scale = (torch.randn(DIM, device=device, dtype=torch.bfloat16) * 0.5 + 1.0).detach().requires_grad_()
residual = torch.randn(B, T, DIM, device=device, dtype=torch.bfloat16).requires_grad_()

ref = residual + mlp_scale * F.linear(x, ste_weight(w))
out = fused_mlp_down_w8a16(x, w, mlp_scale, residual)
cos_fwd = cos(ref, out)

g = torch.randn_like(out)
for t in [x, w, mlp_scale, residual]: t.grad = None
out.backward(g)
out_grads = [x.grad.clone(), w.grad.clone(), mlp_scale.grad.clone(), residual.grad.clone()]
for t in [x, w, mlp_scale, residual]: t.grad = None
ref.backward(g)
cos_gx = cos(x.grad, out_grads[0])
cos_gw = cos(w.grad, out_grads[1])
cos_gs = cos(mlp_scale.grad, out_grads[2])
cos_gr = cos(residual.grad, out_grads[3])
cos_gw = min(cos_gx, cos_gw, cos_gs, cos_gr)

ms_bf = bench(lambda: residual + mlp_scale * F.linear(x, w))
ms_w  = bench(lambda: fused_mlp_down_w8a16(x, w, mlp_scale, residual))
results.append(('MLP_down', (4096, MLP_DIM, DIM), cos_fwd, cos_gx, cos_gw, ms_bf, ms_w))

# ============================================================================
# Raw-kernel-only timing (no autograd bookkeeping)
# ============================================================================
print()
print("Raw-kernel-only timing (pre-quantized weights, no autograd wrapper):")
print(f"{'kernel':>9} | {'ms_bf16_raw':>11} | {'ms_w8a16_raw':>12} | {'ratio':>6}")
print('-' * 52)

x1 = torch.randn(4096, DIM, device=device, dtype=torch.bfloat16)
wq = torch.randn(DIM + 2*KV_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02)
wq_i8, wq_s = quantize_w8a16(wq)
ms_bf = bench(lambda: F.linear(x1, wq))
ms_w  = bench(lambda: _launch_qkv(x1, wq_i8, wq_s))
print(f"{'QKV':>9} | {ms_bf:>10.4f}  | {ms_w:>11.4f}  | {ms_bf/ms_w:>5.2f}x")

wo = torch.randn(DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02)
wo_i8, wo_s = quantize_w8a16(wo)
a_scale = torch.randn(DIM, device=device, dtype=torch.bfloat16).mul_(0.5).add_(1.0)
res = torch.randn(4096, DIM, device=device, dtype=torch.bfloat16)
ms_bf = bench(lambda: res + a_scale * F.linear(x1, wo))
ms_w  = bench(lambda: _launch_out_proj(x1, wo_i8, wo_s, a_scale, res))
print(f"{'OutProj':>9} | {ms_bf:>10.4f}  | {ms_w:>11.4f}  | {ms_bf/ms_w:>5.2f}x")

wu = torch.randn(MLP_DIM, DIM, device=device, dtype=torch.bfloat16).mul_(0.02)
wu_i8, wu_s = quantize_w8a16(wu)
ms_bf = bench(lambda: F.leaky_relu(F.linear(x1, wu), negative_slope=0.5).square())
ms_w  = bench(lambda: _launch_mlp_up(x1, wu_i8, wu_s))
print(f"{'MLP_up':>9} | {ms_bf:>10.4f}  | {ms_w:>11.4f}  | {ms_bf/ms_w:>5.2f}x")

x_mlp = torch.randn(4096, MLP_DIM, device=device, dtype=torch.bfloat16)
wd = torch.randn(DIM, MLP_DIM, device=device, dtype=torch.bfloat16).mul_(0.02)
wd_i8, wd_s = quantize_w8a16(wd)
m_scale = torch.randn(DIM, device=device, dtype=torch.bfloat16).mul_(0.5).add_(1.0)
ms_bf = bench(lambda: res + m_scale * F.linear(x_mlp, wd))
ms_w  = bench(lambda: _launch_mlp_down(x_mlp, wd_i8, wd_s, m_scale, res))
print(f"{'MLP_down':>9} | {ms_bf:>10.4f}  | {ms_w:>11.4f}  | {ms_bf/ms_w:>5.2f}x")

# ============================================================================
# Report
# ============================================================================
print()
print(f"{'kernel':>9} | {'(M,K,N)':>19} | {'cos_fwd':>9} | {'cos_gx':>8} | {'cos_worst':>9} | {'ms_bf16':>8} | {'ms_w8a16':>9} | {'ratio':>6} | {'status':>6}")
print('-' * 110)
pass_all = True
for (name, shape, c_fwd, c_gx, c_worst, ms_bf, ms_w) in results:
    status = 'PASS' if (c_fwd > 0.9995 and c_gx > 0.9995 and c_worst > 0.9995) else 'FAIL'
    if status == 'FAIL': pass_all = False
    print(f"{name:>9} | {str(shape):>19} | {c_fwd:>9.6f} | {c_gx:>8.6f} | {c_worst:>9.6f} | "
          f"{ms_bf:>7.4f} | {ms_w:>8.4f} | {ms_bf/ms_w:>5.2f}x | {status:>6}")

print()
print(f"Overall: {'PASS' if pass_all else 'FAIL'}  (target cos > 0.9995 on forward + all grads)")
