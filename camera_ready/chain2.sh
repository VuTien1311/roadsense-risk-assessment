#!/usr/bin/env bash
# "Maximal" evaluation queue (resumable). Starts when chain.sh has released the GPU.
#  A. remaining 8,400 val images: full system + 10 fusion ablations + 2 detector swaps,
#     all computed online on the same model outputs (no depth cache needed)
#  B. perturbation stability on the 1,000-image random subset (6 perturbations,
#     full + 5 variants, perturbations applied in memory)
#  C. final merge / evaluation / tables / figures / manuscript (run_all.py final)
set -u
cd "$(dirname "$0")/.."
P="${PYTHON:-python}"
R=outputs/camera_ready/runs
S=outputs/camera_ready/sets
LOG=outputs/camera_ready/chain2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for chain.sh GPU stage"
until grep -q "GPU stage done" outputs/camera_ready/chain.log 2>/dev/null; do sleep 120; done

VARIANTS="B0;B1;B2;B3;no_hybridnets;no_depth;single_region;no_scene_quality;no_gates;corridor_road;swap:yolo11s-seg.pt;swap:yolo11m-seg.pt"
log "A: val_rest (8,400 images) with online variants"
until $P camera_ready/harness.py infer --image-list $S/val_rest.txt --out $R/val_rest_full \
      --no-vis --no-depth-cache --resume --variants "$VARIANTS" >> $R/val_rest_full.log 2>&1; do
  log "A interrupted (rc=$?), resuming in 60 s"; sleep 60
done
log "A done: $(ls $R/val_rest_full/upgraded_*.json | wc -l) reports"

PVARIANTS="single_region;no_gates;no_scene_quality;B0;B3"
for PERT in dark bright contrast blur noise jpeg; do
  log "B: standard perturbation $PERT"
  until $P camera_ready/harness.py infer --image-list $S/standard.txt --out $R/stdpert_${PERT}_full \
        --perturb $PERT --no-vis --no-depth-cache --no-yolo-cache --resume --variants "$PVARIANTS" >> $R/stdpert_${PERT}.log 2>&1; do
    log "B $PERT interrupted (rc=$?), resuming in 60 s"; sleep 60
  done
done
log "B done"

log "waiting for chain.sh to finish its CPU stages"
until grep -q "chain finished" outputs/camera_ready/chain.log 2>/dev/null; do sleep 120; done
log "C: final merge, evaluation, tables, figures, manuscript"
$P camera_ready/run_all.py final >> "$LOG" 2>&1
log "chain2 finished"
touch outputs/camera_ready/COMPUTE_DONE
