"""Advisory and reliability-state stability under mild photometric perturbations.

For every hard-set image the clean report and the report of each perturbed copy
(same variant) are compared:

* advisory flip       global advisory differs
* severe flip         advisory moves by >= 2 levels or crosses into/out of
                      BRAKE/STOP
* state flip          for objects matched across the two reports (same class,
                      IoU >= 0.5), the STABLE/CHECK/UNCERTAIN state differs
* object persistence  fraction of clean objects that are matched in the
                      perturbed report
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import json
from pathlib import Path

import numpy as np

from gt_eval import box_iou, load_reports
from stats import cluster_bootstrap_ratio, wilson

SEVERITY = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}
PERTURBATIONS = ("dark", "bright", "contrast", "blur", "noise", "jpeg")


def match_objects(clean, perturbed, threshold=0.5):
    used = set()
    pairs = []
    for i, a in enumerate(clean):
        best_j, best = None, threshold
        for j, b in enumerate(perturbed):
            if j in used or a["class"] != b["class"]:
                continue
            value = box_iou(a["box_xyxy"], b["box_xyxy"])
            if value >= best and (best_j is None or value > best):
                best_j, best = j, value
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def compare_runs(clean_dir, perturbed_dirs, advisory_field="global_advisory", nofallback=False, keys=None):
    clean = load_reports(clean_dir)
    totals = {"images": 0, "flip": 0, "severe_flip": 0}
    state_num, state_den, persist_num, persist_den = [], [], [], []
    per_perturbation = {}
    for name, directory in perturbed_dirs.items():
        reports = load_reports(directory)
        counts = {"images": 0, "flip": 0, "severe_flip": 0}
        for key_pert, report in reports.items():
            key = key_pert.split("__")[0]
            if key not in clean or (keys is not None and key not in keys):
                continue
            base = clean[key]
            a = advisory(base, nofallback)
            b = advisory(report, nofallback)
            counts["images"] += 1
            if a != b:
                counts["flip"] += 1
            if abs(SEVERITY[a] - SEVERITY[b]) >= 2 or ((SEVERITY[a] >= 3) != (SEVERITY[b] >= 3)):
                counts["severe_flip"] += 1
            pairs = match_objects(base["detections"], report["detections"])
            flips = sum(
                1
                for i, j in pairs
                if base["detections"][i]["original_unit_status"] != report["detections"][j]["original_unit_status"]
            )
            state_num.append(flips)
            state_den.append(len(pairs))
            persist_num.append(len(pairs))
            persist_den.append(len(base["detections"]))
        per_perturbation[name] = {
            "images": counts["images"],
            "flip_rate": counts["flip"] / max(counts["images"], 1),
            "severe_flip_rate": counts["severe_flip"] / max(counts["images"], 1),
        }
        for field in totals:
            totals[field] += counts[field]
    flip = wilson(totals["flip"], totals["images"])
    severe = wilson(totals["severe_flip"], totals["images"])
    state = cluster_bootstrap_ratio(state_num, state_den)
    persist = cluster_bootstrap_ratio(persist_num, persist_den)
    return {
        "pairs": totals["images"],
        "flip_rate": flip,
        "severe_flip_rate": severe,
        "state_flip_rate": state,
        "object_persistence": persist,
        "per_perturbation": per_perturbation,
    }


def advisory(report, nofallback):
    if nofallback and report["object_count"] == 0:
        return "CLEAR"
    return report["global_advisory"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Runs directory")
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    runs = Path(args.runs)
    clean_dir = runs / ("hard_full" if args.variant == "full" else f"hard_{args.variant}")
    perturbed = {name: runs / f"pert_{name}_{args.variant}" for name in PERTURBATIONS}
    print(json.dumps(compare_runs(clean_dir, perturbed), indent=2, default=float))


if __name__ == "__main__":
    main()
