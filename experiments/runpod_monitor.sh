#!/bin/bash
# Live training monitor — tails most recent H100 run log, filters key events.
set -eu

RUN_DIR="${RUN_DIR:-/workspace/forgefuse-run}"

# Prefer the native logs/<run_id>.txt which has the full step-by-step output.
LOG=$(ls -t "$RUN_DIR"/logs/h100_phase3a_fullrun_*.txt 2>/dev/null | head -1 || true)
if [ -z "$LOG" ]; then
    LOG=$(ls -t "$RUN_DIR"/logs/h100_phase3a_fullrun_*.console.log 2>/dev/null | head -1 || true)
fi
if [ -z "$LOG" ]; then
    echo "No H100 run log found in $RUN_DIR/logs/"
    exit 1
fi

echo "Monitoring: $LOG"
echo "Watching for: val_bpb, step progress, NaN, Traceback, OOM, DIAGNOSTIC, final_"
echo ""

tail -f "$LOG" | grep -E --line-buffered \
    "val_bpb|^step:|^warmup|NaN|Traceback|Error|OOM|out of memory|Killed|DIAGNOSTIC|final_|stopping_early"
