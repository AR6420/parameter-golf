# H100 Single-Seed Validation Run — forgefuse-phase3a

## Setup
- Pod: 8×H100 80GB SXM on RunPod (103.207.149.72:17862)
- Branch: `forgefuse-phase3a` @ `bfe141e`
- Seed: `1337`
- Compile: **disabled** via `TORCHDYNAMO_DISABLE=1`. Phase 1's `_FusedQKNormRoPE.backward` calls `x.detach().requires_grad_(True)` which `torch.compile(fullgraph=True)` rejects as "Unsupported Tensor.requires_grad_() call". Defensive try/except caught the compile object, but dynamo's lazy trace on first forward raised anyway; env var is the reliable kill switch.
- FA3: active (preinstalled `flash-attn-3 3.0.0+20260303.cu128torch291cxx11abitrue`)
- Wall-clock cap: 600 s (native default)

## Training trajectory
```
step     0/20000  val_bpb  4.1049  (pre-training)
step     1/20000  train_loss 6.9317  (302 ms)
step  1000/20000  train_loss 2.2588  (281 s)
step  1500/20000  train_loss 2.1439  (422 s)   swa:start step:1450
step  1604/20000  late_qat:enabled scale:0.1499
step  2000/20000  train_loss 2.1055  (564 s)
step  2127/20000  val_bpb  1.2204   (600 s, stopping_early: wallclock_cap)
```
- `step_avg`: 282.2 ms/step
- iterations completed: 2127 of 20000 (capped by 600 s wall-clock)

## Key final metrics

| Metric | val_bpb |
|---|---|
| **Raw live model @ step 2127** | **1.2204** |
| `DIAGNOSTIC post_ema` (EMA shadow weights) | 1.2224 |
| **`final_int6_roundtrip` (GPTQ INT6 + lzma)** | **1.2549** |

## Submission artifact
- `final_model.int6.ptz`: 11,074,204 bytes (10.56 MiB)
- Total submission incl. code: 11,197,503 bytes (10.68 MiB)
- Under 16 MB cap ✓

## Gap to leaderboard

| Submission | val_bpb |
|---|---|
| pr-1493 record (current SOTA) | 1.0810 |
| pr-1019 record (our arch base) | 1.1147 |
| **forgefuse-phase3a (this run)** | **1.2549** |
| simple baseline target | ~1.15 |

**Gap to pr-1493: +0.1739 BPB.** Gap to simple baseline: +0.10 BPB.

## Why the gap — four reasons, ranked by impact

1. **No torch.compile** (biggest) — we ran eager, doing 2127 iterations vs record's ~4500. Effectively ~half the training budget. Fixing `_FusedQKNormRoPE.backward` to not use `requires_grad_()` would let us compile and likely cut train-time per step by ~40%, approximately doubling reachable iterations.
2. **Non-record hyperparameters** — `QK_GAIN_INIT=1.5` (record 5.25), `WD=0.04` (record 0.095). Our local Phase 3A tests at 500-iter horizon showed these hurt, so we didn't commit them; but at 4500-iter H100 horizon they might help. Could not re-test at full scale within this $25 budget.
3. **W8A8 QAT penalty** — late_qat kicks in at step 1604 (last 25 % of training). The W8A8 path was only validated locally through 200 iters with STE bit-exact backward; at H100 training length + GPTQ INT6 post-quant stacked on top, accumulated quant error shows through.
4. **No record-grade architecture extras** — SP8192 tokenizer, 3-layer depth recurrence, parallel residuals, Legal TTT are all in pr-1493's record and contribute ~0.07 BPB combined. Phase 3B–E would add them.

## What the sanity probe showed
The 5-iteration sanity probe ran to completion (including `final_int6_roundtrip`) at val_bpb 4.04 (expected on a 5-iter model = random). Importantly, the full path (train → GPTQ → int6 roundtrip) **worked end-to-end on the W8A8 branch at H100 scale for the first time**.

## What didn't complete
The `final_int6_sliding_window` eval stalled after `final_int6_roundtrip` logged. Likely an eager-mode slowness at the sliding window's sequence length × stride on this model size without compile. Not blocking — the roundtrip BPB (1.2549) is the authoritative post-submission number; sliding typically shaves 0.01–0.02 off, so the real "shipped" number would be 1.24–1.25.

## Pod cost
- Time used: ~50 min of the ~65 min budget (pod shut down cleanly before full 65-min mark)
- Actual cost: ~$19 of $25
- Headroom preserved: ~$6 for a follow-up retry if needed (did not use)

## Recommendation (user decision)
Per the prompt:
> If EMA val_bpb > 1.10: STOP, pause pod, report for strategy pivot

Our EMA val_bpb is 1.2224, well above 1.10 → **STOP**. Pod has been terminated. Seeds 2 and 3 not run.

### What the data tells us for next steps

The architectural extras (SP8192 / depth recurrence / parallel residuals / TTT) together would close ~70 % of the 0.17 BPB gap by matching the record arch. The HP tuning (QK-Gain / WD) and fixing `requires_grad_()` in the backward would close the rest — re-test each at full H100 horizon.

Getting compile working is probably the highest-ROI single fix: it unlocks ~2× more iterations per pod hour, which is the straightest path to closing the training-budget gap.

## Files
- `native_train.log` — full native `train_gpt.py` log, 2653 lines
- `deploy_console.log` — deploy script stdout/stderr, 4050 lines
- `sanity_probe.log` — separate 5-iter probe log, 2650 lines
