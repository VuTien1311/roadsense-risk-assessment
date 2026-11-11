"""One entry point that regenerates every camera-ready result from the caches.

Stages (run in this order; each is resumable):

    gpu       perturbed-image inference and detector-swap inference
    priors    dev-split component priors for the no_hybridnets / no_depth variants
    replays   fusion-only replays: regression checks, build-up baselines,
              leave-one-out ablations, dev sensitivity sweeps, perturbed variants
    eval      ground-truth evaluation of every run directory
    summary   metrics, paired comparisons, sensitivity and stability -> results.json
    tables    LaTeX/Markdown tables and figures for the paper

Usage:  python camera_ready/run_all.py gpu replays eval summary tables
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs" / "camera_ready"
RUNS = OUT / "runs"
SETS = OUT / "sets"
PERT = OUT / "perturbed"
CACHE = OUT / "cache"
PERTURBATIONS = ("dark", "bright", "contrast", "blur", "noise", "jpeg")
SWAP_WEIGHTS = ("yolo11s-seg.pt", "yolo11m-seg.pt")
ABLATIONS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "no_hybridnets",
    "no_depth",
    "single_region",
    "no_scene_quality",
    "no_gates",
)
REGRESSION = ("legacy_default", "risk_default", "sq_default")
SENSITIVITY = (
    "sens:stable_c=0.544",
    "sens:stable_c=0.816",
    "sens:stable_a=1.08",
    "sens:stable_a=1.62",
    "sens:check_c=0.28",
    "sens:check_c=0.42",
    "sens:check_a=1.44",
    "sens:check_a=2.16",
    "sens:k_ratio=1.08",
    "sens:k_ratio=1.62",
    "sens:k_spread=0.68",
    "sens:k_spread=1.02",
    "sens:n_support=96",
    "sens:n_support=144",
    "risk:scale=0.8",
    "risk:scale=0.9",
    "risk:scale=1.1",
    "risk:scale=1.2",
    "sq:0.3333,0.3333,0.3334",
    "sq:0.5,0.25,0.25",
    "sq:0.2,0.4,0.4",
)
PERTURBED_VARIANTS = ("single_region", "no_gates", "no_scene_quality", "B0", "B3")
EXPLORATORY = ("corridor_road",)
# Operating curve: all advisory cut-offs scaled together (1.0 = published constants).
OPERATING_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5)
OPERATING_SPLITS = ("dev", "standard")


def operating_variants():
    return [f"risk:scale={scale}" for scale in OPERATING_SCALES]
WORKERS = 4


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def run(command, log=None):
    print("$", " ".join(str(c) for c in command), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as handle:
            return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    return subprocess.run(command, check=False).returncode


def harness(mode, image_list, out, variant="full", weight=None, extra=()):
    command = [PY, str(HERE / "harness.py"), mode, "--image-list", str(image_list), "--out", str(out), "--variant", variant, "--resume"]
    if weight:
        command += ["--yolo-weight", str(ROOT / weight)]
    command += list(extra)
    return command


def stage_gpu():
    for name in PERTURBATIONS:
        out = RUNS / f"pert_{name}_full"
        run(harness("infer", PERT / f"{name}.txt", out, extra=["--no-vis"]), RUNS / f"pert_{name}_full.log")
    for weight in SWAP_WEIGHTS:
        stem = Path(weight).stem.replace("-seg", "")
        for split in ("hard", "standard"):
            out = RUNS / f"{split}_swap_{stem}"
            run(harness("infer", SETS / f"{split}.txt", out, weight=weight, extra=["--no-vis"]), RUNS / f"{split}_swap_{stem}.log")


def stage_priors():
    run([PY, str(HERE / "priors.py")])


def replay_jobs():
    jobs = []
    for variant in REGRESSION:
        jobs.append(("hard", SETS / "hard.txt", RUNS / f"hard_{variant}", variant))
    for split in ("hard", "standard"):
        for variant in ABLATIONS:
            jobs.append((split, SETS / f"{split}.txt", RUNS / f"{split}_{safe(variant)}", variant))
    for variant in SENSITIVITY:
        jobs.append(("dev", SETS / "dev.txt", RUNS / f"dev_{safe(variant)}", variant))
    for split in ("hard", "dev", "standard"):
        for variant in EXPLORATORY:
            jobs.append((split, SETS / f"{split}.txt", RUNS / f"{split}_{safe(variant)}", variant))
    for split in OPERATING_SPLITS:
        for variant in operating_variants():
            if split == "dev" and variant in SENSITIVITY:
                continue
            jobs.append((split, SETS / f"{split}.txt", RUNS / f"{split}_{safe(variant)}", variant))
    for name in PERTURBATIONS:
        for variant in PERTURBED_VARIANTS:
            jobs.append(("pert", PERT / f"{name}.txt", RUNS / f"pert_{name}_{safe(variant)}", variant))
    return jobs


def stage_replays():
    jobs = replay_jobs()

    def work(job):
        _, image_list, out, variant = job
        return run(harness("replay", image_list, out, variant), Path(str(out) + ".log"))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        codes = list(pool.map(work, jobs))
    failed = [str(job[2]) for job, code in zip(jobs, codes) if code != 0]
    print(f"[REPLAYS] {len(jobs)} jobs, failed={len(failed)}")
    for item in failed:
        print("  FAILED", item)
    for variant in REGRESSION:
        run([PY, str(HERE / "harness.py"), "check", "--a", str(RUNS / "hard_full"), "--b", str(RUNS / f"hard_{variant}")])


def eval_targets():
    targets = []
    for split in ("hard", "dev", "standard"):
        targets.append((f"{split}_full", split, "yolo11x-seg"))
    for split in ("hard", "standard"):
        for variant in ABLATIONS:
            targets.append((f"{split}_{safe(variant)}", split, None))
        for weight in SWAP_WEIGHTS:
            stem = Path(weight).stem.replace("-seg", "")
            targets.append((f"{split}_swap_{stem}", split, Path(weight).stem))
    for variant in SENSITIVITY:
        targets.append((f"dev_{safe(variant)}", "dev", None))
    for split in ("hard", "dev", "standard"):
        for variant in EXPLORATORY:
            targets.append((f"{split}_{safe(variant)}", split, None))
    for split in OPERATING_SPLITS:
        for variant in operating_variants():
            if split == "dev" and variant in SENSITIVITY:
                continue
            targets.append((f"{split}_{safe(variant)}", split, None))
    return targets


def yolo_tag_for(stem):
    matches = sorted((CACHE / "yolo").glob(f"{stem}-*"))
    return matches[0].name if matches else ""


def stage_eval():
    def work(target):
        name, split, weight_stem = target
        if not (RUNS / name).exists():
            print("  skip (no run)", name)
            return 0
        command = [PY, str(HERE / "gt_eval.py"), "--run", str(RUNS / name), "--split", split, "--name", name]
        if weight_stem:
            command += ["--yolo-tag", yolo_tag_for(weight_stem)]
        if name.endswith("_full"):
            command += ["--corridor"]
        return run(command)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        codes = list(pool.map(work, eval_targets()))
    if any(codes):
        raise SystemExit(f"gt_eval failed for {sum(1 for c in codes if c)} runs")


def must(command):
    """Run one step of the summary / tables / final stages; a failed step stops everything after it
    (a silently failed results.py once left a stale results.json behind)."""
    code = run(command)
    if code != 0:
        raise SystemExit(f"step failed (exit {code}): {' '.join(str(c) for c in command)}")


def stage_summary():
    must([PY, str(HERE / "corridor_stats.py")])
    must([PY, str(HERE / "results.py")])


def stage_tables():
    must([PY, str(HERE / "make_tables.py")])
    must([PY, str(HERE / "figures.py")])


TEST_VARIANTS = ("full",) + ABLATIONS + EXPLORATORY + ("swap_yolo11s", "swap_yolo11m")


def stage_final():
    """Merge the 9,500-image test split, evaluate it, and rebuild every artefact."""
    must([PY, str(HERE / "merge_test.py")])

    def work(variant):
        name = f"test_{safe(variant)}"
        command = [PY, str(HERE / "gt_eval.py"), "--run", str(RUNS / name), "--split", "test", "--name", name]
        if variant == "full":
            command += ["--yolo-tag", yolo_tag_for("yolo11x-seg"), "--corridor"]
        return run(command)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        codes = list(pool.map(work, TEST_VARIANTS))
    if any(codes):
        raise SystemExit(f"gt_eval failed for {sum(1 for c in codes if c)} test runs")
    must([PY, str(HERE / "audit_kit.py"), "unsafe"])
    stage_summary()
    stage_tables()
    must([PY, str(HERE / "paper_numbers.py")])


STAGES = {
    "gpu": stage_gpu,
    "priors": stage_priors,
    "replays": stage_replays,
    "eval": stage_eval,
    "summary": stage_summary,
    "tables": stage_tables,
    "final": stage_final,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        print(f"===== {stage} =====", flush=True)
        STAGES[stage]()


if __name__ == "__main__":
    main()
