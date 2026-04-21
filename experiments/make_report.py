"""Build tables + 6-curve plot + COMPARISON_REPORT.md from JSON outputs."""
import json, os
from pathlib import Path

EXP_DIR = Path('experiments')
BRANCHES = [
    ('pr-1019',   'Old simple baseline (9L/2x)',      'pr1019'),
    ('pr-1493',   'Current simple baseline (9L/2x)',  'pr1493'),
    ('preker',    'Pre-kernels ctrl (11L/3x+XSA/VE)', 'preker'),
    ('sub2',      'Sub #2 control (+Triton kernels)', 'sub2'),
    ('w8a16',     'ForgeFuse W8A16',                  'w8a16'),
    ('w8a8',      'ForgeFuse W8A8',                   'w8a8'),
]


def load_json(tag, phase):
    p = EXP_DIR / f'exp_{phase}_{tag}.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def bpb_at(data, step_or_sec, by='step'):
    if not data or not data.get('checkpoints'):
        return None
    for c in data['checkpoints']:
        key = c['step'] if by == 'step' else c['wallclock']
        target = step_or_sec
        # tolerant match: within 5% of target
        if abs(key - target) / max(1, target) < 0.1:
            return c['val_bpb']
    return None


def loss_at_step(data, step):
    if not data or not data.get('losses'):
        return None
    idx = None
    for i, s in enumerate(data['step_indices']):
        if s == step:
            idx = i; break
    return data['losses'][idx] if idx is not None else None


# ========== Experiment A table ==========
print("Experiment A table (500 steps)")
rows_a = []
for tag, label, key in BRANCHES:
    d = load_json(key, 'a')
    if d is None:
        continue
    rows_a.append({
        'tag': tag, 'label': label, 'key': key,
        'n_params': d['n_params'],
        'loss_10':  loss_at_step(d, 10),
        'loss_100': loss_at_step(d, 100),
        'loss_250': loss_at_step(d, 250),
        'loss_500': loss_at_step(d, 500),
        'bpb_100':  bpb_at(d, 100),
        'bpb_250':  bpb_at(d, 250),
        'bpb_500':  bpb_at(d, 500),
        'step_ms':  d.get('step_ms_est'),
        'peak_vram': d.get('peak_vram_mb'),
        'total_wc': d['total_wallclock'],
    })

# Write exp_a_equal_steps.txt
lines_a = []
lines_a.append("Experiment A — Equal steps (500), BT=8192, AdamW, seed 1337, bf16 autocast")
lines_a.append("")
lines_a.append(f"{'Branch':>10} | {'Step 10':>9} | {'Step 100':>9} | {'Step 250':>9} | {'Step 500':>9}")
lines_a.append(f"{'':>10} | {'(loss)':>9} | {'(loss)':>9} | {'(loss)':>9} | {'(loss)':>9}")
lines_a.append('-' * 70)
for r in rows_a:
    def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else ' n/a '
    lines_a.append(f"{r['tag']:>10} | {fmt(r['loss_10']):>9} | {fmt(r['loss_100']):>9} | {fmt(r['loss_250']):>9} | {fmt(r['loss_500']):>9}")
lines_a.append('')
lines_a.append('val_bpb @ step 500:')
bpb500 = {r['tag']: r['bpb_500'] for r in rows_a}
for r in rows_a:
    lines_a.append(f"  {r['label']:<40}  {r['bpb_500']:.4f}" if r['bpb_500'] else f"  {r['label']:<40}  n/a")
lines_a.append('')
lines_a.append('Isolated deltas (val_bpb @ step 500):')
def d(a,b): return (bpb500.get(a) or 0) - (bpb500.get(b) or 0) if bpb500.get(a) and bpb500.get(b) else None
deltas = [
    ('pr-1019 vs pr-1493 (time evolution of simple baseline)', 'pr-1019', 'pr-1493'),
    ('pr-1019 vs pre-kernels ctrl (our arch work value)',       'pr-1019', 'preker'),
    ('pre-kernels vs Sub#2 ctrl (our kernels value)',           'preker',  'sub2'),
    ('Sub#2 vs W8A16 (W8A16 quality cost)',                     'sub2',    'w8a16'),
    ('W8A16 vs W8A8 (W8A8 additional cost)',                    'w8a16',   'w8a8'),
]
for desc, a, b in deltas:
    val = d(a, b)
    sign = '+' if val and val > 0 else ''
    lines_a.append(f"  {desc}:  {sign}{val:.4f}" if val is not None else f"  {desc}:  n/a")
lines_a.append('')
lines_a.append('Step time (ms) and peak VRAM (MB):')
for r in rows_a:
    sm = f"{r['step_ms']:.0f}" if r['step_ms'] else 'n/a'
    vm = f"{r['peak_vram']:.0f}" if r['peak_vram'] else 'n/a'
    lines_a.append(f"  {r['label']:<40}  {sm} ms/step, peak {vm} MB")

with open(EXP_DIR / 'exp_a_equal_steps.txt', 'w') as f:
    f.write('\n'.join(lines_a))
print('\n'.join(lines_a))

# ========== Experiment B table ==========
print("\nExperiment B table (600s wall-clock)")
rows_b = []
for tag, label, key in BRANCHES:
    d = load_json(key, 'b')
    if d is None:
        continue
    rows_b.append({
        'tag': tag, 'label': label, 'key': key,
        'total_steps': d['total_steps'],
        'bpb_120': bpb_at(d, 120, by='wallclock'),
        'bpb_300': bpb_at(d, 300, by='wallclock'),
        'bpb_600': bpb_at(d, 600, by='wallclock'),
    })

lines_b = []
lines_b.append("Experiment B — Equal wall-clock (600s), BT=8192, AdamW, seed 1337")
lines_b.append("")
lines_b.append(f"{'Branch':>10} | {'Steps':>6} | {'2min BPB':>9} | {'5min BPB':>9} | {'10min BPB':>10}")
lines_b.append('-' * 60)
for r in rows_b:
    def fmt(x): return f"{x:.4f}" if x is not None else ' n/a '
    lines_b.append(f"{r['tag']:>10} | {r['total_steps']:>6} | {fmt(r['bpb_120']):>9} | {fmt(r['bpb_300']):>9} | {fmt(r['bpb_600']):>10}")

with open(EXP_DIR / 'exp_b_equal_wallclock.txt', 'w') as f:
    f.write('\n'.join(lines_b))
print('\n'.join(lines_b))

# ========== Plot ==========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = {'pr-1019': '#999999', 'pr-1493': '#666666',
          'preker': '#3366CC', 'sub2': '#22AA22',
          'w8a16': '#FF8800', 'w8a8': '#CC0000'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for tag, label, key in BRANCHES:
    d = load_json(key, 'b')
    if d is None: continue
    wcs = d['wallclock']; losses = d['losses']
    ax1.plot(wcs, losses, label=label, color=colors.get(tag, 'black'), alpha=0.85, linewidth=1.2)
    # BPB from checkpoints
    if d.get('checkpoints'):
        xs = [c['wallclock'] for c in d['checkpoints']]
        ys = [c['val_bpb'] for c in d['checkpoints']]
        ax2.plot(xs, ys, marker='o', label=label, color=colors.get(tag, 'black'), linewidth=1.8)
ax1.set_xlabel('Wall-clock (s)'); ax1.set_ylabel('Train loss')
ax1.set_title('Train loss vs wall-clock (600s each)')
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax2.set_xlabel('Wall-clock (s)'); ax2.set_ylabel('val_bpb')
ax2.set_title('val_bpb checkpoints (120, 300, 600s)')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig.suptitle('ForgeFuse local comparison (RTX 5070 Ti, SDPA math fallback, BT=8192, AdamW)')
fig.tight_layout()
fig.savefig(EXP_DIR / 'exp_b_curves.png', dpi=130)
print(f"\nSaved plot: {EXP_DIR / 'exp_b_curves.png'}")
