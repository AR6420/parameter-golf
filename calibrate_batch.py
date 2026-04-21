"""Batch-size calibration for local head-to-head training runs.

Works across the 3 branches we compare: gemm-boundary-fusion-attn, pr-1493,
forgefuse-w8a8 (and -w8a16). Auto-detects which kwargs the branch's GPT class
accepts and supplies only those, so one script runs on all three.

For each candidate (B, T) it:
  1. Builds the model in bf16 autocast
  2. Runs 1 warmup + 1 measured forward + backward pass
  3. Reports peak VRAM in MB
"""
import os, sys, types, inspect, math, argparse, traceback
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import torch
import torch.nn.functional as F

# Mock flash_attn — all 3 branches expect it
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

# Kitchen-sink config; filtered per-branch below
FULL_CFG = dict(
    vocab_size=1024,
    num_layers=11,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3,
    tie_embeddings=True,
    tied_embed_init_std=0.02,
    logit_softcap=30.0,
    rope_base=1024.0,
    qk_gain_init=1.0,
    rope_dims=16,
)

def build_model():
    sig = inspect.signature(train_gpt.GPT.__init__)
    accepted = set(sig.parameters.keys())
    kw = {k: v for k, v in FULL_CFG.items() if k in accepted}
    missing = [p.name for p in sig.parameters.values()
               if p.name not in ('self', 'args', 'kwargs')
               and p.default is inspect.Parameter.empty
               and p.name not in kw]
    if missing:
        raise RuntimeError(f"GPT __init__ requires args we don't have: {missing}")
    m = train_gpt.GPT(**kw).cuda()
    # Float bank weights if this branch has them
    for name in ('qkv_bank', 'out_bank', 'qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank'):
        p = getattr(m, name, None)
        if p is not None:
            p.data = p.data.float()
    return m, kw

def try_batch(m, B, T):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(1337)
        ids = torch.randint(0, FULL_CFG['vocab_size'], (B, T), device='cuda')
        tgt = torch.randint(0, FULL_CFG['vocab_size'], (B, T), device='cuda')
        # warmup
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(ids, tgt)
        loss.backward()
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # measure
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(ids, tgt)
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        step_ms = start.elapsed_time(end)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        m.zero_grad(set_to_none=True)
        return dict(ok=True, peak_mb=peak_mb, step_ms=step_ms, loss=loss.item())
    except torch.cuda.OutOfMemoryError as e:
        return dict(ok=False, err='OOM', msg=str(e)[:120])
    except Exception as e:
        return dict(ok=False, err=type(e).__name__, msg=str(e)[:150], tb=traceback.format_exc()[-400:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='?')
    ap.add_argument('--candidates', default='4096,8192,16384,32768,65536')
    ap.add_argument('--seq-len', type=int, default=1024)
    ap.add_argument('--num-layers', type=int, default=11)
    ap.add_argument('--mlp-mult', type=int, default=3)
    args = ap.parse_args()

    FULL_CFG['num_layers'] = args.num_layers
    FULL_CFG['mlp_mult'] = args.mlp_mult

    print(f"=== Calibration: {args.label} ===")
    try:
        m, cfg = build_model()
    except Exception as e:
        print(f"Model build FAILED: {e}")
        return
    print(f"Config used: num_layers={cfg.get('num_layers','?')}, model_dim={cfg.get('model_dim','?')}, mlp_mult={cfg.get('mlp_mult','?')}")
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total params: {total_params/1e6:.1f}M")

    T = args.seq_len
    print(f"{'batch_tokens':>13} | {'B':>4} | {'T':>5} | {'peak_MB':>9} | {'step_ms':>8} | status")
    print('-' * 62)
    for bt_str in args.candidates.split(','):
        bt = int(bt_str)
        if bt % T != 0:
            B = max(1, bt // T); T_use = bt // B
        else:
            B = bt // T; T_use = T
        r = try_batch(m, B, T_use)
        if r['ok']:
            print(f"{bt:>13} | {B:>4} | {T_use:>5} | {r['peak_mb']:>9.1f} | {r['step_ms']:>8.2f} | OK")
        else:
            print(f"{bt:>13} | {B:>4} | {T_use:>5} | {'-':>9} | {'-':>8} | FAIL ({r['err']}) {r['msg']}")
    del m
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
