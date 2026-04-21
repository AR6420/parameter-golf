"""
Submission #2 — Local Validation Script
Runs on RTX 5070 Ti (no FA3, no torch.compile).
Tests kernel correctness, speedup, gradients, memory, and 200-step loss match.
"""
import os, sys, math, time, types
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'

# Mock flash_attn before importing train_gpt
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

import train_gpt
from train_gpt import (
    fused_mlp_up_proj, fused_mlp_down_proj,
    _triton_mlp_up_proj_fwd, _triton_mlp_down_proj_fwd,
    MLP, Block, RMSNorm,
)
from torch import Tensor, nn

device = 'cuda'
torch.manual_seed(1337)

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: KERNEL ISOLATION BENCHMARK
# ═══════════════════════════════════════════════════════════════════════
print("=" * 68)
print("SECTION 1: KERNEL ISOLATION BENCHMARK")
print("=" * 68)

M, K_up, N_up = 4096, 512, 1536
K_dn, N_dn = 1536, 512
WARMUP, ITERS = 20, 500

x_up = torch.randn(M, K_up, device=device, dtype=torch.bfloat16)
w_up = torch.randn(N_up, K_up, device=device, dtype=torch.bfloat16)
x_dn = torch.randn(M, K_dn, device=device, dtype=torch.bfloat16)
w_dn = torch.randn(N_dn, K_dn, device=device, dtype=torch.bfloat16)
scale = torch.randn(N_dn, device=device, dtype=torch.bfloat16)
resid = torch.randn(M, N_dn, device=device, dtype=torch.bfloat16)

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

# --- Kernel 1 ---
ref_up = F.leaky_relu(F.linear(x_up, w_up), negative_slope=0.5).square()
out_up = _triton_mlp_up_proj_fwd(x_up, w_up)
cos_up = F.cosine_similarity(ref_up.flatten().float(), out_up.flatten().float(), dim=0).item()

ms_cublas_up = bench(lambda: F.leaky_relu(F.linear(x_up, w_up), negative_slope=0.5).square())
ms_triton_up = bench(lambda: _triton_mlp_up_proj_fwd(x_up, w_up))
speedup_up = ms_cublas_up / ms_triton_up

print(f"\nKernel 1 -- fused_mlp_up_proj [{M}, {K_up}] @ [{K_up}, {N_up}]:")
print(f"  cuBLAS + separate act:  {ms_cublas_up:.4f} ms")
print(f"  Triton fused:           {ms_triton_up:.4f} ms")
print(f"  Speedup:                {speedup_up:.2f}x")
print(f"  cos_sim:                {cos_up:.6f}")

# --- Kernel 2 ---
ref_dn = resid + scale[None, :] * F.linear(x_dn, w_dn)
out_dn = _triton_mlp_down_proj_fwd(x_dn, w_dn, scale, resid)
cos_dn = F.cosine_similarity(ref_dn.flatten().float(), out_dn.flatten().float(), dim=0).item()

ms_cublas_dn = bench(lambda: resid + scale[None, :] * F.linear(x_dn, w_dn))
ms_triton_dn = bench(lambda: _triton_mlp_down_proj_fwd(x_dn, w_dn, scale, resid))
speedup_dn = ms_cublas_dn / ms_triton_dn

print(f"\nKernel 2 -- fused_mlp_down_proj [{M}, {K_dn}] @ [{K_dn}, {N_dn}]:")
print(f"  cuBLAS + separate ops:  {ms_cublas_dn:.4f} ms")
print(f"  Triton fused:           {ms_triton_dn:.4f} ms")
print(f"  Speedup:                {speedup_dn:.2f}x")
print(f"  cos_sim:                {cos_dn:.6f}")

save_up = ms_cublas_up - ms_triton_up
save_dn = ms_cublas_dn - ms_triton_dn
total_save = (save_up + save_dn) * 11  # 11 blocks, 1 up + 1 down each
print(f"\nCombined MLP savings per step (11 blocks on H100):")
print(f"  Up proj:   {save_up:.4f} ms x 11 = {save_up*11:.2f} ms")
print(f"  Down proj: {save_dn:.4f} ms x 11 = {save_dn*11:.2f} ms")
print(f"  Total:     {total_save:.2f} ms / 86.7ms baseline = {total_save/86.7*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: IN-MODEL CORRECTNESS (200 STEPS)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("SECTION 2: IN-MODEL CORRECTNESS (200 steps)")
print("=" * 68)

# Build a standalone MLP block that mirrors the SOTA Block's MLP path.
# Skip attention (too slow with SDPA math) — just test the MLP fusion.
MODEL_DIM = 512
MLP_DIM = 1536
B, T = 2, 128  # small batch, enough to exercise the path
N_STEPS = 200

class MLPBlock(nn.Module):
    """Isolated MLP block: RMSNorm -> MLP -> scale + residual."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale_factor = 1.0 / math.sqrt(6)
        # Simulate parameter banks (fp32 weights)
        self.up_w = nn.Parameter(torch.randn(int(dim * mlp_mult), dim) * 0.02)
        self.down_w = nn.Parameter(torch.randn(dim, int(dim * mlp_mult)) * 0.02)
    def forward(self, x):
        normed = self.norm(x) * self.ln_scale_factor
        return self.mlp(normed, self.up_w, self.down_w,
                        mlp_scale=self.mlp_scale.to(x.dtype), residual=x)

# Run 200 steps in both modes, collect losses at checkpoints
checkpoints = {1, 10, 50, 100, 200}
losses_baseline = {}
losses_fused = {}

for mode_name, use_triton in [("baseline", False), ("fused", True)]:
    train_gpt._USE_TRITON_MLP = use_triton
    torch.manual_seed(42)
    model = MLPBlock(MODEL_DIM, 3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = losses_baseline if mode_name == "baseline" else losses_fused

    for step in range(1, N_STEPS + 1):
        torch.manual_seed(1000 + step)
        x = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16)
        target = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16)
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

train_gpt._USE_TRITON_MLP = True  # restore

print(f"\n{'Step':>6} | {'Loss (baseline)':>15} | {'Loss (fused)':>14} | {'Diff':>10}")
print("-" * 56)
max_diff = 0.0
for step in sorted(checkpoints):
    lb = losses_baseline[step]
    lf = losses_fused[step]
    diff = abs(lb - lf)
    max_diff = max(max_diff, diff)
    print(f"{step:>6} | {lb:>15.6f} | {lf:>14.6f} | {diff:>10.6f}")
print(f"\nMax loss diff across {N_STEPS} steps: {max_diff:.6f}")
loss_pass = max_diff < 0.01

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: MEMORY DELTA
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("SECTION 3: MEMORY DELTA")
print("=" * 68)

mem_results = {}
for mode_name, use_triton in [("baseline", False), ("fused", True)]:
    train_gpt._USE_TRITON_MLP = use_triton
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    model = MLPBlock(MODEL_DIM, 3.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16)
    target = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16)
    opt.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    opt.step()

    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    mem_results[mode_name] = peak
    del model, opt, x, target, out, loss
    torch.cuda.empty_cache()

print(f"\n  Baseline (_USE_TRITON=False): {mem_results['baseline']:.1f} MiB")
print(f"  Fused    (_USE_TRITON=True):  {mem_results['fused']:.1f} MiB")
delta = mem_results['fused'] - mem_results['baseline']
pct = delta / mem_results['baseline'] * 100
print(f"  Delta: {delta:+.1f} MiB ({pct:+.1f}%)")

train_gpt._USE_TRITON_MLP = True

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: GRADIENT NORM CHECK
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("SECTION 4: GRADIENT NORM CHECK (10 steps)")
print("=" * 68)

grad_norms = {"baseline": [], "fused": []}
for mode_name, use_triton in [("baseline", False), ("fused", True)]:
    train_gpt._USE_TRITON_MLP = use_triton
    torch.manual_seed(42)
    model = MLPBlock(MODEL_DIM, 3.0).to(device)

    for step in range(10):
        torch.manual_seed(1000 + step)
        x = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randn(B, T, MODEL_DIM, device=device, dtype=torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        gn = x.grad.norm().item()
        grad_norms[mode_name].append(gn)
        model.zero_grad()

    del model
    torch.cuda.empty_cache()

train_gpt._USE_TRITON_MLP = True

print(f"\n{'Step':>5} | {'GradNorm (baseline)':>20} | {'GradNorm (fused)':>17} | {'Match?':>7}")
print("-" * 60)
max_grad_pct = 0.0
all_grad_pass = True
for i in range(10):
    gb = grad_norms["baseline"][i]
    gf = grad_norms["fused"][i]
    pct_diff = abs(gb - gf) / (gb + 1e-12) * 100
    max_grad_pct = max(max_grad_pct, pct_diff)
    ok = pct_diff < 1.0
    if not ok:
        all_grad_pass = False
    print(f"  {i+1:>3} | {gb:>20.6f} | {gf:>17.6f} | {'YES' if ok else 'NO':>6}")
print(f"\nMax grad norm deviation: {max_grad_pct:.4f}%")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════
k1_pass = speedup_up >= 1.3
k2_pass = speedup_dn >= 1.0
grad_pass = all_grad_pass
overall = k1_pass and k2_pass and loss_pass and grad_pass

print("\n")
print("+====================================================+")
print("|     SUBMISSION #2 LOCAL VALIDATION REPORT           |")
print("+====================================================+")
print(f"| Kernel 1 speedup (M=4096):    {speedup_up:.2f}x    {'[PASS]' if k1_pass else '[FAIL]':>8} |")
print(f"| Kernel 2 speedup (M=4096):    {speedup_dn:.2f}x    {'[PASS]' if k2_pass else '[FAIL]':>8} |")
print(f"| Loss diff @ 200 steps:        {max_diff:.6f}  {'[PASS]' if loss_pass else '[FAIL]':>8} |")
print(f"| Grad norm match (10 steps):   {max_grad_pct:.2f}%    {'[PASS]' if grad_pass else '[FAIL]':>8} |")
print(f"| Memory delta:                 {delta:+.0f} MiB   [INFO]   |")
print(f"| Combined per-step savings:    {total_save:.2f}ms   [INFO]   |")
print("+====================================================+")
print(f"| OVERALL: {'READY FOR H100' if overall else 'NOT READY':^39}|")
print("+====================================================+")

if not overall:
    fails = []
    if not k1_pass: fails.append("Kernel 1 speedup < 1.3x")
    if not k2_pass: fails.append("Kernel 2 speedup < 1.0x")
    if not loss_pass: fails.append(f"Loss diff {max_diff:.6f} >= 0.01")
    if not grad_pass: fails.append(f"Grad norm deviation {max_grad_pct:.2f}% >= 1%")
    print(f"\nFAILED: {', '.join(fails)}")
