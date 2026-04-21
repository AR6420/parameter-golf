"""Unified 6-branch comparison harness.

One script, run after each `git checkout <branch>` + optional flag edits.
The current train_gpt.py on disk defines the GPT class; we build it with a
filtered FULL_CFG (inspect.signature picks only args the branch accepts).

All 6 runs use:
  - same seed (1337)
  - same data shard (fineweb_train_000000.bin, fineweb_val_000000.bin)
  - same batch config (B=8, T=1024 -> BT=8192)
  - same AdamW optimizer (NOT each branch's native Muon, to isolate
    architecture/quant differences from optimizer differences)
  - same bf16 autocast
  - same val_bpb formula

Two modes:
  --mode steps   --num-steps 500       (Experiment A, equal steps)
  --mode seconds --seconds 600         (Experiment B, equal wall-clock)
"""
import os, sys, types, inspect, argparse, json, math, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CC'] = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe'
import numpy as np
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

FULL_CFG = dict(
    vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
    logit_softcap=30.0, rope_base=1024.0, qk_gain_init=1.0, rope_dims=16,
)
SEED = 1337
B, T = 8, 1024

DATA_TRAIN = './data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin'
DATA_VAL = './data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'
TOKENIZER = './data/tokenizers/fineweb_1024_bpe.model'


def load_shard(path):
    data = np.fromfile(path, dtype=np.uint16)
    # fineweb .bin header: 256 uint16 at the start, tokens follow
    return torch.from_numpy(data[256:].astype(np.int32))


def compute_bytes_per_token(tokens):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    sample = tokens[:1_000_000].tolist()
    text = sp.decode(sample)
    return len(text.encode('utf-8')) / len(sample)


def build_model(cfg):
    sig = inspect.signature(train_gpt.GPT.__init__)
    accepted = set(sig.parameters.keys())
    kw = {k: v for k, v in cfg.items() if k in accepted}
    missing = [p.name for p in sig.parameters.values()
               if p.name not in ('self', 'args', 'kwargs')
               and p.default is inspect.Parameter.empty and p.name not in kw]
    if missing:
        raise RuntimeError(f"GPT __init__ missing: {missing}")
    m = train_gpt.GPT(**kw).cuda()
    for name in ('qkv_bank', 'out_bank', 'qo_bank', 'kv_bank', 'mlp_up_bank', 'mlp_down_bank'):
        p = getattr(m, name, None)
        if p is not None:
            p.data = p.data.float()
    return m, kw


def sample_batch(tokens, B, T, rng):
    starts = rng.integers(0, tokens.numel() - T - 1, size=(B,), dtype=np.int64)
    ids = torch.stack([tokens[s:s + T + 1] for s in starts]).long().cuda()
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


@torch.no_grad()
def measure_val_bpb(model, val_tokens, bpt, n_batches=16, B=4, T=1024):
    """Mean CE loss on val shard -> BPB = (nll / ln 2) / bytes_per_token."""
    model.eval()
    losses = []
    rng = np.random.default_rng(999)
    for _ in range(n_batches):
        starts = rng.integers(0, val_tokens.numel() - T - 1, size=(B,), dtype=np.int64)
        ids = torch.stack([val_tokens[s:s + T + 1] for s in starts]).long().cuda()
        x = ids[:, :-1].contiguous(); y = ids[:, 1:].contiguous()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, y)
        losses.append(float(loss))
    model.train()
    mean_nll = sum(losses) / len(losses)
    bpb = (mean_nll / math.log(2)) / bpt
    return mean_nll, bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['steps', 'seconds'], required=True)
    ap.add_argument('--num-steps', type=int, default=500)
    ap.add_argument('--seconds', type=float, default=600.0)
    ap.add_argument('--num-layers', type=int, default=11)
    ap.add_argument('--mlp-mult', type=int, default=3)
    ap.add_argument('--checkpoint-steps', default='100,250,500')
    ap.add_argument('--checkpoint-seconds', default='120,300,600')
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-3)
    args = ap.parse_args()

    FULL_CFG['num_layers'] = args.num_layers
    FULL_CFG['mlp_mult'] = args.mlp_mult

    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"[{args.label}] Loading data...")
    train_tokens = load_shard(DATA_TRAIN)
    val_tokens = load_shard(DATA_VAL)
    print(f"  train={train_tokens.numel()/1e6:.1f}M tokens, val={val_tokens.numel()/1e6:.1f}M tokens")

    bpt = compute_bytes_per_token(val_tokens)
    print(f"  bytes/token = {bpt:.4f}")

    print(f"[{args.label}] Building model...")
    model, cfg = build_model(FULL_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params, cfg={cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    rng = np.random.default_rng(SEED)
    cp_steps = set(int(x) for x in args.checkpoint_steps.split(',')) if args.mode == 'steps' else set()
    cp_secs = sorted(float(x) for x in args.checkpoint_seconds.split(',')) if args.mode == 'seconds' else []

    result = dict(label=args.label, cfg=cfg, n_params=n_params, bpt=bpt, mode=args.mode,
                  lr=args.lr, seed=SEED, batch_B=B, batch_T=T,
                  losses=[], grad_norms=[], wallclock=[], step_indices=[],
                  checkpoints=[])

    # Prime (compile kernels) before wallclock starts counting
    x0, y0 = sample_batch(train_tokens, B, T, np.random.default_rng(SEED))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss0 = model(x0, y0)
    loss0.backward(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print(f"[{args.label}] Training ({args.mode}) — checkpoints at {sorted(cp_steps) or cp_secs}")
    t_start = time.perf_counter()
    step = 0
    loss_val = None
    try:
        while True:
            if args.mode == 'steps' and step >= args.num_steps: break
            if args.mode == 'seconds' and (time.perf_counter() - t_start) >= args.seconds: break

            step += 1
            ids, tgt = sample_batch(train_tokens, B, T, rng)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(ids, tgt)
            loss.backward()
            gn2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn2 += float(p.grad.float().norm()) ** 2
            gn = math.sqrt(gn2)
            loss_val = float(loss)
            if math.isnan(loss_val) or math.isnan(gn):
                print(f"  [NaN at step {step}] ABORT")
                result['nan_at_step'] = step
                break
            opt.step()
            wc = time.perf_counter() - t_start
            if step % args.log_every == 0:
                result['losses'].append(loss_val)
                result['grad_norms'].append(gn)
                result['wallclock'].append(wc)
                result['step_indices'].append(step)
            do_cp = False
            if args.mode == 'steps' and step in cp_steps: do_cp = True
            if args.mode == 'seconds' and cp_secs and wc >= cp_secs[0]:
                do_cp = True
                cp_secs = cp_secs[1:]
            if do_cp:
                nll, bpb = measure_val_bpb(model, val_tokens, bpt)
                result['checkpoints'].append(dict(step=step, wallclock=wc, loss=loss_val,
                                                  val_nll=nll, val_bpb=bpb))
                print(f"  step {step:>4} | wc {wc:6.1f}s | loss {loss_val:.4f} | gn {gn:.3f} | val_bpb {bpb:.4f}")
                with open(args.output, 'w') as f: json.dump(result, f, indent=2)
            elif step % 50 == 0:
                print(f"  step {step:>4} | wc {wc:6.1f}s | loss {loss_val:.4f} | gn {gn:.3f}")
        if not result['checkpoints'] or result['checkpoints'][-1]['step'] != step:
            nll, bpb = measure_val_bpb(model, val_tokens, bpt)
            result['checkpoints'].append(dict(step=step, wallclock=time.perf_counter() - t_start,
                                              loss=loss_val, val_nll=nll, val_bpb=bpb))
    except Exception as e:
        import traceback
        result['error'] = str(e)[:300]
        result['traceback'] = traceback.format_exc()[-500:]
        print(f"  EXCEPTION: {e}")

    result['total_steps'] = step
    result['total_wallclock'] = time.perf_counter() - t_start
    if len(result['wallclock']) >= 2:
        dt = result['wallclock'][-1] - result['wallclock'][0]
        dsteps = result['step_indices'][-1] - result['step_indices'][0]
        result['step_ms_est'] = (dt / max(1, dsteps)) * 1000 if dsteps else None
    else:
        result['step_ms_est'] = None
    result['peak_vram_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[{args.label}] DONE. steps={step}, wc={result['total_wallclock']:.1f}s, "
          f"step_ms~{result.get('step_ms_est')}, peak_vram={result['peak_vram_mb']:.0f} MB, "
          f"checkpoints={len(result['checkpoints'])}")
    print(f"  Output: {args.output}")

    del model, opt, train_tokens, val_tokens
    gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
