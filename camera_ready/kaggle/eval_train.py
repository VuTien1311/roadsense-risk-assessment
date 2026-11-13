#!/usr/bin/env python3
"""Evaluate the train-split runs against the BDD100K train ground truth (Kaggle CPU kernel).

Inputs (attached): roadscene-cr-train-bundle (code), the roadscene-cr-verify output (train
ground truth in the gt_eval format, the checked key list), the drivable masks dataset, and the
outputs of the train GPU sessions (T_c<chunk>_s<shard>.zip; the team account's sessions as
<session>__T_c<chunk>_s<shard>.zipdata in the roadscene-cr-train-team-out dataset). The same gt_eval.py as for the
headline results is used, with --labels pointing at the train ground truth and --yolo-boxes at
the compact detector outputs.

Outputs (/kaggle/working): eval_train.zip (eval/<run>/images.csv, objects.csv for every run),
eval_status.json (images per run, duplicates, missing).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()
WORK = Path("/tmp/eval_work")
OUT = Path("/kaggle/working")
RUNS = ["train_full", "train_no_hybridnets", "train_risk_scale_0.8", "train_risk_scale_1.2",
        "train_risk_scale_1.4", "train_risk_scale_1.6", "train_risk_scale_2.0", "train_no_gates", "train_B0"]
# Peak memory of one gt_eval run on the 69,863 train images: 0.9 GB measured on the 9,500 test
# images (reports and labels scale with the image count) -> about 6-7 GB; 8 GB leaves a margin.
PER_RUN_GB = 8.0


def log(message):
    print(f"[{time.time() - T0:7.0f} s] {message}", flush=True)


PINS = ["numpy==2.4.6", "opencv-python==4.13.0.92", "opencv-python-headless==4.13.0.92", "pillow==12.2.0",
        "ultralytics==8.4.66", "timm==1.0.27", "efficientnet_pytorch==0.7.1", "pretrainedmodels==0.7.4",
        "matplotlib==3.10.9", "prefetch_generator==1.0.3", "webcolors==25.10.0"]


def memory_available_gb():
    """Memory this kernel can still use. Inside a container /proc/meminfo shows the host, so when a
    cgroup limit is set, the limit minus the cgroup's usage (without reclaimable page cache) counts."""
    found = {}
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        found["meminfo_available"] = int(info["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        pass
    for folder, limit_name, usage_name in ((Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
                                           (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes",
                                            "memory.usage_in_bytes")):
        try:
            limit = (folder / limit_name).read_text().strip()
            usage = int((folder / usage_name).read_text().strip())
        except (OSError, ValueError):
            continue
        if limit.isdigit() and int(limit) < 1 << 60:
            try:
                stat = dict(line.split() for line in (folder / "memory.stat").read_text().splitlines()
                            if len(line.split()) == 2)
                usage -= int(stat.get("inactive_file", stat.get("total_inactive_file", 0)))
            except (OSError, ValueError):
                pass
            found["cgroup_limit"], found["cgroup_available"] = int(limit), int(limit) - usage
        break
    usable = [found[k] for k in ("meminfo_available", "cgroup_available") if k in found]
    return (round(min(usable) / 1e9, 1) if usable else None), {k: round(v / 1e9, 1) for k, v in found.items()}


def worker_count(available_gb, per_run_gb=PER_RUN_GB, reserve_gb=3.0, most=4):
    """gt_eval keeps every report of its run in memory and parses the whole train samples.json, so
    on the 69,863 train images one run needs several GB; run only as many at once as fit."""
    if available_gb is None:
        return 1
    return max(1, min(most, int((available_gb - reserve_gb) // per_run_gb)))


def max_child_rss_gb():
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)  # kB on Linux


def install():
    """gt_eval imports test3 (for the ego corridor), which imports torch and ultralytics."""
    py = sys.executable
    subprocess.run([py, "-m", "pip", "uninstall", "-y", "-q", "opencv-contrib-python", "opencv-contrib-python-headless"])
    subprocess.run([py, "-m", "pip", "install", "-q", "--no-cache-dir", "torch==2.7.1", "torchvision==0.22.1",
                    "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run([py, "-m", "pip", "install", "-q", "--no-cache-dir", *PINS], check=True)
    check = subprocess.run([py, "-c", "import cv2, numpy, torch, ultralytics; print(cv2.__version__, numpy.__version__, "
                            "torch.__version__, ultralytics.__version__)"], capture_output=True, text=True)
    log(f"environment: {check.stdout.strip()} {check.stderr.strip()[-300:]}")
    return check.stdout.strip()


def main():
    status = {"environment": install()}
    code = Path(sorted(glob.glob("/kaggle/input/**/roadscene-cr-train-bundle/code/test3.py", recursive=True))[0]).parent
    report = Path(sorted(glob.glob("/kaggle/input/**/verify_report.json", recursive=True))[0])
    verified = json.loads(report.read_text())
    if not verified.get("all_checks_passed"):
        raise SystemExit("verification did not pass")
    masks = Path(verified["train"]["mask_dir"])
    labels = WORK / "train_labels"
    (labels / "fields").mkdir(parents=True, exist_ok=True)
    os.symlink(report.parent / "train_labels" / "samples.json", labels / "samples.json")
    os.symlink(masks, labels / "fields" / "drivable")

    # Union of all train chunks of all sessions; a key finished twice keeps its first copy.
    runs, cache = WORK / "runs", WORK / "cache" / "hybridnets"
    cache.mkdir(parents=True, exist_ok=True)
    boxes, duplicates = [], 0
    # Kernel outputs of the main account's sessions (T_c*_s*.zip) and the team account's session
    # archives, copied into a dataset as <session>__T_c*_s*.zipdata so Kaggle does not unpack them.
    archives = sorted(glob.glob("/kaggle/input/**/T_c*_s*.zip", recursive=True)
                      + glob.glob("/kaggle/input/**/*T_c*_s*.zipdata", recursive=True))
    unpacked = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            unpacked += sum(info.file_size for info in handle.infolist())
    free = shutil.disk_usage(WORK).free
    status.update({"unpacked_gb": round(unpacked / 1e9, 1), "disk_free_gb": round(free / 1e9, 1)})
    log(f"{len(archives)} archives, {unpacked / 1e9:.1f} GB unpacked, {free / 1e9:.1f} GB free under {WORK}")
    if free < 1.1 * unpacked + 2e9:
        raise SystemExit("not enough disk space to unpack the archives")
    for index, archive in enumerate(archives):
        target = WORK / "chunks" / f"{index:03d}"
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
        for run in RUNS:
            (runs / run).mkdir(parents=True, exist_ok=True)
            for path in (target / "runs" / run).glob("upgraded_*.json"):
                destination = runs / run / path.name
                if destination.exists():
                    duplicates += 1
                else:
                    path.rename(destination)
        for path in (target / "cache" / "hybridnets").glob("*.npz"):
            destination = cache / path.name
            if not destination.exists():
                path.rename(destination)
        if (target / "boxes.jsonl").exists():
            boxes.append(str(target / "boxes.jsonl"))
    per_run = {run: {p.stem.replace("upgraded_", "", 1) for p in (runs / run).glob("upgraded_*.json")} for run in RUNS}
    complete = sorted(set.intersection(*per_run.values())) if per_run else []
    keys = (report.parent / "train_keys.txt").read_text().split()
    sets = WORK / "sets"
    sets.mkdir(parents=True, exist_ok=True)
    (sets / "train_ids.txt").write_text("".join(f"{k}\n" for k in complete))
    status.update({"archives": len(archives), "duplicates": duplicates, "train_keys": len(keys),
                   "complete_in_all_runs": len(complete), "per_run": {r: len(v) for r, v in per_run.items()},
                   "box_files": len(boxes)})
    log(json.dumps(status))

    def evaluate(run):
        command = [sys.executable, str(code / "camera_ready" / "gt_eval.py"), "--run", str(runs / run), "--split", "train",
                   "--name", run, "--labels", str(labels), "--sets-dir", str(sets), "--eval-dir", str(WORK / "eval"),
                   "--cache", str(WORK / "cache")]
        if run == "train_full":
            command += ["--yolo-boxes", ",".join(boxes), "--corridor"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=code)
        log(f"{run}: rc={result.returncode} {result.stdout.strip()[-300:]} {result.stderr.strip()[-500:]}")
        return run, result.returncode

    available, memory = memory_available_gb()
    workers = worker_count(available)
    status.update({"memory_gb": memory, "gt_eval_workers": workers})
    log(f"memory {memory} GB -> {workers} gt_eval runs in parallel (about {PER_RUN_GB:g} GB each)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        status["gt_eval_returncodes"] = dict(pool.map(evaluate, RUNS))
    status["max_gt_eval_rss_gb"] = max_child_rss_gb()
    with zipfile.ZipFile(OUT / "eval_train.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted((WORK / "eval").rglob("*.csv")):
            handle.write(path, path.relative_to(WORK).as_posix())
        handle.write(sets / "train_ids.txt", "sets/train_ids.txt")
    (OUT / "eval_status.json").write_text(json.dumps(status, indent=1))
    log("finished")


if __name__ == "__main__":
    main()
