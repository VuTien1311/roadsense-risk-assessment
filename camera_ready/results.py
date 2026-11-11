"""Collect every camera-ready metric into outputs/camera_ready/results.json.

Splits: ``hard`` (100, stress), ``test`` (9,500 = BDD100K val minus dev; all
headline numbers), ``standard`` (1,000-image random subset of test; perturbation
stability), ``dev`` (500; priors and threshold sensitivity only).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import json
import statistics
from pathlib import Path

import numpy as np

from run_all import ABLATIONS, EXPLORATORY, PERTURBATIONS, PERTURBED_VARIANTS, RUNS, SENSITIVITY, SWAP_WEIGHTS, safe
from stability import compare_runs
from summarize import EVAL_DIR, compare, load_eval, summarize, write_json

OUT = Path(__file__).resolve().parents[1] / "outputs" / "camera_ready"
ABLATION_SPLITS = ("hard", "standard", "test", "train")


def exists(name):
    return (EVAL_DIR / name / "images.csv").exists()


def latency(run_names):
    rows = []
    for name in run_names:
        path = RUNS / f"{name}_timing.jsonl"
        if path.exists():
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            # Only images whose three networks actually ran in this process (no cache hits), after warm-up.
            rows += [row for row in items if row.get("index", 0) > 1 and {"depth_s", "hybridnets_s", "yolo_s"} <= set(row)]
    if not rows:
        return None
    result = {"images": len(rows)}
    for field in ("hybridnets_s", "depth_s", "yolo_s", "total_s"):
        values = [row[field] for row in rows]
        result[field] = {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p90": float(np.percentile(values, 90)),
        }
    result["peak_vram_mb"] = max(row.get("peak_vram_mb", 0.0) for row in rows)
    return result


def subset_changes(split, base):
    """Advisory changes of the source-removal ablations, by time of day."""
    base_rows, _ = load_eval(base)
    out = {}
    for variant in ("no_depth", "no_hybridnets"):
        name = f"{split}_{variant}"
        if not exists(name):
            continue
        other = {row["key"]: row for row in load_eval(name)[0]}
        for group in ("daytime", "night"):
            rows = [row for row in base_rows if row["timeofday"] == group and row["key"] in other]
            changed = sum(1 for row in rows if row["advisory"] != other[row["key"]]["advisory"])
            out[f"{split}:{variant}:{group}"] = {"images": len(rows), "changed": changed,
                                                 "fraction": changed / len(rows) if rows else float("nan")}
    return out


def operating_curve():
    """Unsafe CLEAR vs over-caution when all advisory cut-offs are scaled together."""
    from run_all import OPERATING_SCALES, OPERATING_SPLITS

    curve = {}
    for split in tuple(OPERATING_SPLITS) + ("train", "valop"):
        points = []
        for scale in sorted(set(OPERATING_SCALES) | {1.0}):
            name = f"{split}_full" if scale == 1.0 else f"{split}_{safe(f'risk:scale={scale}')}"
            if not exists(name):
                continue
            rows, _ = load_eval(name)
            occupied = [r for r in rows if r["occupied"] == "1"]
            free = [r for r in rows if r["occupied_any"] == "0"]
            severity = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}
            points.append(
                {
                    "scale": scale,
                    "images": len(rows),
                    "occupied": len(occupied),
                    "unsafe_clear": sum(r["advisory"] == "CLEAR" for r in occupied),
                    "free": len(free),
                    "over_caution": sum(severity[r["advisory"]] >= 2 for r in free),
                    "clear_images": sum(r["advisory"] == "CLEAR" for r in rows),
                    "severe_images": sum(r["advisory"] in {"BRAKE", "STOP"} for r in rows),
                }
            )
        if points:
            curve[split] = points
    return curve


def stability_block(clean_prefix, perturbed_prefix):
    block = {}
    for variant in ("full",) + PERTURBED_VARIANTS:
        clean = RUNS / f"{clean_prefix}_{safe(variant)}"
        perturbed = {name: RUNS / f"{perturbed_prefix}_{name}_{safe(variant)}" for name in PERTURBATIONS}
        if clean.exists() and all(path.exists() and any(path.glob("upgraded_*.json")) for path in perturbed.values()):
            block[variant] = compare_runs(clean, perturbed, nofallback=variant.startswith("B"))
    return block


def crosshw_block():
    """Same images, code and software versions, two GPUs: local RTX 3050 vs Kaggle T4."""
    block = {}
    for variant in ("full",) + PERTURBED_VARIANTS:
        local, other = RUNS / f"standard_{safe(variant)}", RUNS / f"stdkaggle_{safe(variant)}"
        if local.exists() and any(other.glob("upgraded_*.json")):
            block[variant] = compare_runs(local, {"t4": other}, nofallback=variant.startswith("B"))
    # val_rest images processed locally (not the ones imported from the Kaggle run) vs their T4 rerun in valop_full.
    listing = RUNS / "val_rest_kaggle_keys.txt"
    if any((RUNS / "valop_full").glob("upgraded_*.json")) and listing.exists():
        on_t4 = set(listing.read_text(encoding="utf-8").split())
        local_keys = {p.stem.replace("upgraded_", "", 1) for p in (RUNS / "val_rest_full").glob("upgraded_*.json")} - on_t4
        block["val_rest_full"] = compare_runs(RUNS / "val_rest_full", {"t4": RUNS / "valop_full"}, keys=local_keys)
    return block


def kaggle_info():
    listing = RUNS / "val_rest_kaggle_keys.txt"
    provenance = OUT / "kaggle" / "provenance.json"
    if not listing.exists():
        return {}
    info = {"test_images_on_t4": sum(1 for line in listing.read_text(encoding="utf-8").splitlines() if line.strip())}
    if provenance.exists():
        data = json.loads(provenance.read_text(encoding="utf-8"))
        info["kaggle_env"] = data.get("kaggle_env")
        info["local_env"] = data.get("local_env")
    return info


def main():
    results = {
        "splits": {},
        "ablations": {},
        "comparisons": {},
        "sensitivity": {},
        "stability": {},
        "stability_std": {},
        "swap": {},
        "latency": {},
        "exploratory": {},
        "subset_changes": {},
    }
    for split in ("hard", "dev", "standard", "test", "train", "valop"):
        if exists(f"{split}_full"):
            results["splits"][split] = summarize(f"{split}_full")
    results["latency"]["all"] = latency(["hard_full", "dev_full", "standard_full", "val_rest_full"])

    for split in ABLATION_SPLITS:
        base = f"{split}_full"
        if not exists(base):
            continue
        entries = {"full": results["splits"].get(split) or summarize(base)}
        for variant in ABLATIONS:
            name = f"{split}_{safe(variant)}"
            if not exists(name):
                continue
            entries[variant] = summarize(name)
            column = "advisory_nofb" if variant.startswith("B") else "advisory"
            results["comparisons"][f"{split}:full_vs_{variant}"] = compare(base, name, "advisory", column)
        results["comparisons"][f"{split}:full_vs_no_fallback"] = compare(base, base, "advisory", "advisory_nofb")
        results["ablations"][split] = entries
        results["subset_changes"].update(subset_changes(split, base))

    if exists("dev_full"):
        rows = {"default": results["splits"].get("dev") or summarize("dev_full")}
        for variant in SENSITIVITY:
            name = f"dev_{safe(variant)}"
            if exists(name):
                rows[variant] = summarize(name)
                results["comparisons"][f"dev:full_vs_{variant}"] = compare("dev_full", name)
        results["sensitivity"] = rows

    # Hard set: perturbed copies were written to disk (runs/pert_<p>_<variant>);
    # 1,000-image subset: perturbations applied in memory (runs/stdpert_<p>_<variant>).
    results["stability"] = stability_block("hard", "pert")
    # The perturbed 1k runs were made on a Kaggle T4. Compare them with the clean 1k run made on
    # the same hardware (stdkaggle_*) when it exists, so hardware differences never count as flips.
    std_clean = "stdkaggle" if any((RUNS / "stdkaggle_full").glob("upgraded_*.json")) else "standard"
    results["stability_std"] = stability_block(std_clean, "stdpert")
    results["stability_std_reference"] = std_clean
    results["crosshw"] = crosshw_block()
    results["kaggle"] = kaggle_info()
    results["latency"]["t4"] = latency(["t4bench_full"])

    for split in ("hard", "dev", "standard", "test"):
        for variant in EXPLORATORY:
            name = f"{split}_{safe(variant)}"
            if exists(name) and exists(f"{split}_full"):
                results["exploratory"][f"{split}:{variant}"] = summarize(name)
                results["comparisons"][f"{split}:full_vs_{variant}"] = compare(f"{split}_full", name)

    results["operating"] = operating_curve()

    for weight in SWAP_WEIGHTS:
        stem = Path(weight).stem.replace("-seg", "")
        for split in ("hard", "standard", "test"):
            name = f"{split}_swap_{stem}"
            if exists(name):
                results["swap"][f"{split}:{stem}"] = summarize(name)

    write_json(OUT / "results.json", results)
    print(f"[RESULTS] wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
