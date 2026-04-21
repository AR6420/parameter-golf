#!/bin/bash
# RunPod 8xH100-SXM deployment for forgefuse-phase3a
# Target: full-budget training run + final BPB measurement on native train_gpt.py
set -euxo pipefail

# ── Config ────────────────────────────────────────────────────────────────
BRANCH="forgefuse-phase3a"
REPO="https://github.com/AR6420/parameter-golf.git"
RUN_DIR="/workspace/forgefuse-run"
TRAIN_SHARDS=${TRAIN_SHARDS:-10}

# ── Clone ─────────────────────────────────────────────────────────────────
cd /workspace
rm -rf "$RUN_DIR"
git clone -b "$BRANCH" "$REPO" "$RUN_DIR"
cd "$RUN_DIR"

# Verify correct commit
git log --oneline -3
git branch --show-current
echo "Expected HEAD: Phase 3A commit (EMA 0.9965, matrix_lr 0.022, VAL_MAX_BATCHES, compiled_model->model fix)"

# ── Environment verification ──────────────────────────────────────────────
pip list 2>/dev/null | grep -iE "^(torch|triton|flash|sentencepiece|numpy)" || true
python -c "import torch; print(f'torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}')"
python -c "import triton; print(f'triton {triton.__version__}')"
python -c "from flash_attn_interface import flash_attn_func; print('FA3 import OK')" || {
    echo "WARN: FA3 not importable — installing..."
    pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ || {
        echo "ERR: FA3 install failed. Pod must have FA3 for competitive timing."
        exit 1
    }
}

nvidia-smi | head -20
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Data ──────────────────────────────────────────────────────────────────
# Canonical downloader: data/cached_challenge_fineweb.py (not download_data.py)
if [ ! -f "./data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" ]; then
    echo "Downloading FineWeb sp1024 (${TRAIN_SHARDS} train shards)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
else
    echo "FineWeb already present, skipping download."
fi
ls -la data/datasets/fineweb10B_sp1024/ | head -12

# ── Run ───────────────────────────────────────────────────────────────────
# Leaving ITERATIONS, TRAIN_BATCH_TOKENS, MAX_WALLCLOCK_SECONDS at native defaults
# so the run follows the record-like protocol (self-terminates at 600s or iters).
# VAL_MAX_BATCHES=0 disables our local dev subsample — full val pass on H100.
export VAL_MAX_BATCHES=0
export SEED=${SEED:-1337}
export RUN_ID="h100_phase3a_fullrun_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs
LOG_FILE="logs/${RUN_ID}.console.log"

# ── 5-step sanity probe ───────────────────────────────────────────────────
# Catches catastrophic setup failures (bad imports, device mismatches, NaN at
# init, fullgraph compile rejection without fallback) in ~90 s before we
# commit to the 10-min training run. Cheap insurance.
echo ""
echo "=========================================="
echo "5-STEP SANITY PROBE (catches setup issues)"
echo "=========================================="
PROBE_LOG="logs/sanity_probe_$(date +%Y%m%d_%H%M%S).log"
NUM_ITERATIONS=5 ITERATIONS=5 VAL_LOSS_EVERY=5 SEED=1337 VAL_MAX_BATCHES=50 \
    RUN_ID="sanity_probe_$(date +%Y%m%d_%H%M%S)" \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$PROBE_LOG" || true

if grep -qE "NaN|Traceback|RuntimeError|AcceleratorError|out of memory" "$PROBE_LOG"; then
    echo "SANITY PROBE FAILED — error detected. Investigate before full run."
    grep -E "NaN|Traceback|RuntimeError|AcceleratorError|out of memory" "$PROBE_LOG" | head -20
    exit 1
fi

if ! grep -qE "^step:5/5" "$PROBE_LOG"; then
    echo "SANITY PROBE did not reach step 5 — aborting."
    tail -40 "$PROBE_LOG"
    exit 1
fi

# Emit the compile-mode summary from the probe so we know what we're running
echo ""
echo "--- COMPILE MODES (from sanity probe) ---"
grep -E "\[compile\]" "$PROBE_LOG" | head -10
echo ""
echo "SANITY PROBE passed. Starting full run."

# ── Full run ──────────────────────────────────────────────────────────────
echo "==========================================" | tee -a "$LOG_FILE"
echo "Launching torchrun nproc_per_node=8"      | tee -a "$LOG_FILE"
echo "Branch: $BRANCH  RUN_ID: $RUN_ID"         | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee -a "$LOG_FILE"

# ── Results extraction ────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL RESULTS ($RUN_ID)"
echo "=========================================="
grep -E "^step:.*val_bpb" "logs/${RUN_ID}.txt" 2>/dev/null | tail -10 || \
    grep -E "val_bpb" "$LOG_FILE" | tail -15

echo ""
echo "--- DIAGNOSTIC / FINAL BPB LINES ---"
grep -E "DIAGNOSTIC|final_int|final_|post_ema" "logs/${RUN_ID}.txt" 2>/dev/null | tail -15 || \
    grep -E "DIAGNOSTIC|final_int|final_|post_ema" "$LOG_FILE" | tail -15

echo ""
echo "--- LAST 60 LINES OF CONSOLE LOG ---"
tail -60 "$LOG_FILE"

# Save a compact final snapshot for off-pod review
cp "$LOG_FILE" "logs/${RUN_ID}.snapshot.log" || true
if [ -f "logs/${RUN_ID}.txt" ]; then
    cp "logs/${RUN_ID}.txt" "logs/${RUN_ID}.full.log"
fi
echo "Snapshots: logs/${RUN_ID}.snapshot.log  logs/${RUN_ID}.full.log"
