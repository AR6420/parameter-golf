"""ForgeFuse Phase 1 QKV micro-benchmark.
Measures: 3 separate F.linear vs 1 fused F.linear at [4096, 512].
Also extracts prior ForgeFuse phase savings from local_validate_submission2.py run for combined summary."""
import os
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch, torch.nn.functional as F

device = 'cuda'
torch.manual_seed(1337)

# SOTA shapes: B*T=4096, model_dim=512, num_heads=8, num_kv_heads=4, head_dim=64
M, K = 4096, 512
Q_DIM, KV_DIM = 512, 256       # 8*64, 4*64
QKV_DIM = Q_DIM + 2 * KV_DIM   # 1024

x  = torch.randn(M, K,       device=device, dtype=torch.bfloat16)
qw = torch.randn(Q_DIM, K,   device=device, dtype=torch.bfloat16)
kw = torch.randn(KV_DIM, K,  device=device, dtype=torch.bfloat16)
vw = torch.randn(KV_DIM, K,  device=device, dtype=torch.bfloat16)
qkv_w = torch.cat([qw, kw, vw], dim=0).contiguous()

WARMUP, ITERS = 50, 1000

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS

def unfused():
    q = F.linear(x, qw)
    k = F.linear(x, kw)
    v = F.linear(x, vw)
    return q, k, v

def fused():
    qkv = F.linear(x, qkv_w)
    return qkv.split([Q_DIM, KV_DIM, KV_DIM], dim=-1)

# Correctness spot-check
q_u, k_u, v_u = unfused()
q_f, k_f, v_f = fused()
cos_q = F.cosine_similarity(q_u.flatten().float(), q_f.flatten().float(), dim=0).item()
cos_k = F.cosine_similarity(k_u.flatten().float(), k_f.flatten().float(), dim=0).item()
cos_v = F.cosine_similarity(v_u.flatten().float(), v_f.flatten().float(), dim=0).item()
print(f"cos_sim  q={cos_q:.6f}  k={cos_k:.6f}  v={cos_v:.6f}")

ms_unfused = bench(unfused)
ms_fused   = bench(fused)
speedup = ms_unfused / ms_fused
savings_per_block = ms_unfused - ms_fused
N_BLOCKS = 11  # SOTA architecture

print()
print(f"QKV projection at [M={M}, K={K}] -> Q[{Q_DIM}] + K[{KV_DIM}] + V[{KV_DIM}]:")
print(f"  3 separate F.linear (baseline):    {ms_unfused:.4f} ms")
print(f"  1 fused F.linear (forgefuse):      {ms_fused:.4f} ms")
print(f"  Speedup:                           {speedup:.2f}x")
print(f"  Savings per block:                 {savings_per_block:.4f} ms")
print(f"  Savings per step ({N_BLOCKS} attn blocks): {savings_per_block * N_BLOCKS:.3f} ms")

# Dump for combined summary
import json
with open('forgefuse_qkv_bench.json', 'w') as f:
    json.dump({
        'ms_unfused': ms_unfused, 'ms_fused': ms_fused, 'speedup': speedup,
        'savings_per_block_ms': savings_per_block,
        'savings_per_step_ms': savings_per_block * N_BLOCKS,
        'n_blocks': N_BLOCKS, 'M': M, 'K': K, 'Q_DIM': Q_DIM, 'KV_DIM': KV_DIM,
        'cos_q': cos_q, 'cos_k': cos_k, 'cos_v': cos_v,
    }, f, indent=2)
