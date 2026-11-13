"""Merge the results of the Kaggle run (run_kaggle.py) into the local run directories.

    python camera_ready/kaggle/merge.py --src <folder with the downloaded out/ files>           (check only)
    python camera_ready/kaggle/merge.py --src <folder> --apply                                  (import)

Rules:
* A local report is never overwritten.
* A val_rest image is imported only when all 13 variant reports exist for it on Kaggle and
  none exists locally. Its HybridNets and YOLO caches are imported with it, so the reports
  and the caches that gt_eval reads come from the same run; a leftover local cache file from
  the interrupted local run is moved to ``cache_quarantine/`` (never deleted).
* Report "image" paths are rewritten to the local image paths (as the local runs store them).
* Kaggle timing files keep their own names (``runs/kaggle_val_rest_s<k>_timing.jsonl``,
  ``runs/t4bench_full_timing.jsonl``), so the published latency stays on one GPU.
* The clean subset run on Kaggle goes to ``runs/stdkaggle_<variant>`` and the perturbed runs
  to ``runs/stdpert_<perturbation>_<variant>``; results.py compares them on the same hardware.

Writes ``outputs/camera_ready/kaggle/provenance.json`` and ``runs/val_rest_kaggle_keys.txt``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "camera_ready"
RUNS = OUT / "runs"
CACHE = OUT / "cache"
STAGE = OUT / "kaggle" / "staging"
VAL_REST = (
    "full", "B0", "B1", "B2", "B3", "no_hybridnets", "no_depth", "single_region",
    "no_scene_quality", "no_gates", "corridor_road", "swap_yolo11s", "swap_yolo11m",
)
SUBSET = ("full", "single_region", "no_gates", "no_scene_quality", "B0", "B3")
PERTURBATIONS = ("dark", "bright", "contrast", "blur", "noise", "jpeg")
YOLO_TAG = "yolo11x-seg-56f78e29"


def read_list(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem(name):
    return name.replace("upgraded_", "", 1)


def finished(directory):
    return {stem(p.stem) for p in directory.glob("upgraded_*.json")} if directory.exists() else set()


def extract(src, stage):
    jobs = {}
    for archive in sorted(Path(src).glob("*_s[0-9].zip")):
        target = stage / archive.stem
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            if partial.exists():
                shutil.rmtree(partial)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(partial)
            partial.rename(target)
        jobs[archive.stem] = target
    return jobs


def complete_keys(runs, names):
    sets = []
    for name in names:
        directory = runs / name
        good = set()
        for path in directory.glob("upgraded_*.json") if directory.exists() else []:
            try:
                json.loads(path.read_text(encoding="utf-8"))
                good.add(stem(path.stem))
            except (OSError, ValueError):
                print("  unreadable report:", path)
        sets.append(good)
    return set.intersection(*sets) if sets else set()


def copy_report(source, target, local_path, apply):
    if target.exists():
        return False
    if apply:
        report = json.loads(source.read_text(encoding="utf-8"))
        report["image"] = local_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return True


def import_val_rest(jobs, paths, apply, notes):
    local = set().union(*(finished(RUNS / f"val_rest_{name}") for name in VAL_REST))
    imported = []
    for job, root in sorted(jobs.items()):
        if not job.startswith("A_s"):
            continue
        runs = root / "runs"
        for key in sorted(complete_keys(runs, [f"val_rest_{name}" for name in VAL_REST])):
            if key in local:
                notes["val_rest_already_local"].append(key)
                continue
            for name in VAL_REST:
                source = runs / f"val_rest_{name}" / f"upgraded_{key}.json"
                copy_report(source, RUNS / f"val_rest_{name}" / source.name, paths[key], apply)
            for sub, suffix in (("hybridnets", ".npz"), (f"yolo/{YOLO_TAG}", ".pkl")):
                source = root / "cache" / sub / f"{key}{suffix}"
                if not source.exists():
                    notes["missing_cache"].append(f"{sub}/{key}")
                    continue
                target = CACHE / sub / source.name
                if apply:
                    if target.exists():
                        quarantine = OUT / "cache_quarantine" / sub / source.name
                        quarantine.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(target), str(quarantine))
                        notes["quarantined_cache"].append(f"{sub}/{key}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            imported.append(key)
        timing = runs / "val_rest_full_timing.jsonl"
        if apply and timing.exists():
            shutil.copy2(timing, RUNS / f"kaggle_val_rest_{job[-2:]}_timing.jsonl")
    return imported


def import_subset(jobs, job_prefix, run_prefix, paths, apply):
    keys_all = set()
    for job, root in sorted(jobs.items()):
        if not job.startswith(f"{job_prefix}_s"):
            continue
        runs = root / "runs"
        keys = complete_keys(runs, [f"{run_prefix}_{name}" for name in SUBSET])
        for key in sorted(keys):
            for name in SUBSET:
                source = runs / f"{run_prefix}_{name}" / f"upgraded_{key}.json"
                copy_report(source, RUNS / f"{run_prefix}_{name}" / source.name, paths[key], apply)
        keys_all |= keys
    return keys_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="folder with the downloaded Kaggle out/ files")
    parser.add_argument("--apply", action="store_true", help="copy into runs/ and cache/ (default: check only)")
    args = parser.parse_args()
    src = Path(args.src)
    status = json.loads((src / "status.json").read_text(encoding="utf-8"))
    if status.get("problems") or status.get("hash_failures"):
        raise SystemExit(f"the Kaggle run reported problems: {status.get('problems')} {status.get('hash_failures')}")
    if status.get("smoke"):
        print(f"NOTE: smoke run (SMOKE={status['smoke']}); use it for checks only, not for --apply")
        if args.apply:
            raise SystemExit("refusing to import a smoke run")
    stage = STAGE / (src.parent.name if src.name == "out" else src.name)
    stage.mkdir(parents=True, exist_ok=True)
    jobs = extract(src, stage)
    val_rest = read_list(OUT / "sets" / "val_rest.txt")
    standard = read_list(OUT / "sets" / "standard.txt")
    paths = {Path(p).stem: p for p in val_rest + standard}
    local_done = finished(RUNS / "val_rest_swap_yolo11m")
    manifest_keys = [Path(p).stem for p in val_rest if Path(p).stem not in local_done]

    notes = {"val_rest_already_local": [], "missing_cache": [], "quarantined_cache": []}
    imported = import_val_rest(jobs, paths, args.apply, notes)
    clean = import_subset(jobs, "C", "stdkaggle", paths, args.apply)
    perturbed = {name: import_subset(jobs, f"B_{name}", f"stdpert_{name}", paths, args.apply) for name in PERTURBATIONS}
    bench = jobs.get("bench_s0")
    if args.apply and bench and (bench / "runs" / "t4bench_full_timing.jsonl").exists():
        shutil.copy2(bench / "runs" / "t4bench_full_timing.jsonl", RUNS / "t4bench_full_timing.jsonl")

    still_missing = sorted(set(manifest_keys) - set(imported) - set(notes["val_rest_already_local"]))
    summary = {
        "applied": args.apply,
        "source": str(src),
        "kaggle_env": status.get("env"),
        "local_env": status.get("local_env"),
        "jobs": status.get("jobs"),
        "val_rest_imported": len(imported),
        "val_rest_still_missing": len(still_missing),
        "subset_clean_images": len(clean),
        "subset_perturbed_images": {name: len(keys) for name, keys in perturbed.items()},
        "notes": {key: value[:50] for key, value in notes.items()},
    }
    print(json.dumps({k: v for k, v in summary.items() if k not in ("jobs", "kaggle_env", "local_env")}, indent=1))
    if args.apply:
        (OUT / "kaggle").mkdir(parents=True, exist_ok=True)
        (OUT / "kaggle" / "provenance.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        listing = RUNS / "val_rest_kaggle_keys.txt"
        previous = set(read_list(listing)) if listing.exists() else set()
        listing.write_text("".join(f"{key}\n" for key in sorted(previous | set(imported))), encoding="utf-8", newline="\n")
        (OUT / "kaggle" / "val_rest_still_missing.txt").write_text(
            "".join(f"{key}\n" for key in still_missing), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
