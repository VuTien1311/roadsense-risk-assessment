#!/usr/bin/env bash
# Runs the remaining camera-ready stages in dependency order (resumable).
set -u
cd "$(dirname "$0")/.."
P="${PYTHON:-python}"
R=outputs/camera_ready/runs
S=outputs/camera_ready/sets
LOG=outputs/camera_ready/chain.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for standard_full (1000 reports)"
while [ "$(ls $R/standard_full/upgraded_*.json 2>/dev/null | wc -l)" -lt 1000 ]; do sleep 60; done
log "standard_full complete"

log "GPU stage (perturbed + detector swap) started"
$P camera_ready/run_all.py gpu >> "$LOG" 2>&1 &
GPU_PID=$!

log "standard replays started"
for V in B0 B1 B2 B3 no_hybridnets no_depth single_region no_scene_quality no_gates corridor_road; do
  echo "$V"
done | xargs -P 3 -I{} bash -c "$P camera_ready/harness.py replay --image-list $S/standard.txt --out $R/standard_{} --variant {} --resume > $R/standard_{}.log 2>&1; echo 'standard {} rc='\$? >> $LOG"
log "standard replays done"

wait $GPU_PID
log "GPU stage done"

log "remaining replays (perturbed variants; finished jobs are skipped)"
$P camera_ready/run_all.py replays >> "$LOG" 2>&1
log "evaluation"
$P camera_ready/run_all.py eval >> "$LOG" 2>&1
log "summary, tables, figures"
$P camera_ready/run_all.py summary tables >> "$LOG" 2>&1
$P camera_ready/paper_numbers.py >> "$LOG" 2>&1
log "chain finished"
