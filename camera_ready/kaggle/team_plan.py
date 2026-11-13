"""Which train keys are still free for new sessions (e.g. on the team account), reconstructed exactly.

Every train session (run_kaggle.py, MODE "train") works on a key list fixed by its settings:
TRAIN_ASSIGN (an explicit list) or TRAIN_BASE/TRAIN_PART over the keys its predecessors (kernel
sources) had not finished. Each of its GPU shards takes every other key of that list and works
through it in chunks of TRAIN_CHUNK images, strictly in order, stopping only between images
(harness.py --stop-at). So the keys a COMPLETED session finished are, in every chunk, the first
`reports` keys of the chunk (status.json), and a RUNNING session can at most reach the first
budget_h * MAX_RATE keys of each shard list (rounded up to whole chunks).

    python camera_ready/kaggle/team_plan.py --sessions roadscene-cr-train-1 roadscene-cr-train-2 \
        roadscene-cr-train-3 --running roadscene-cr-train-2 roadscene-cr-train-3 --split 2 --out DIR

writes DIR/assign_<i>.txt (the free keys split into --split interleaved parts) and DIR/plan.json.
Consistency checks (status this_part / done_before against the reconstruction, no chunk ending
early without the deadline) stop the plan instead of guessing.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
KERNELS = WORKSPACE / "tmp" / "kaggle_kernels"
DOWNLOADS = WORKSPACE / "tmp" / "kaggle_out"
KEYS_FILE = DOWNLOADS / "roadscene-cr-verify" / "train_keys.txt"
CHUNK = 2000
# Images per hour per GPU: train-1 took 1.88-1.97 h per 2,000-image chunk (1,018-1,064 / h);
# 1,270 / h is 20 % above the mean, used only to bound what a running session can reach.
MAX_RATE = 1270.0


def constant(source, name, default):
    match = re.search(rf"^{name} = (.*?)(  #.*)?$", source, flags=re.M)
    return default if match is None else ast.literal_eval(match.group(1))  # literals written by make_kernel.py


def session_config(slug):
    folder = KERNELS / slug
    source = (folder / "run_kaggle.py").read_text(encoding="utf-8")
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = {
        "slug": slug,
        "owner": metadata["id"].split("/", 1)[0],
        "mode": constant(source, "MODE", "val"),
        "budget_h": float(constant(source, "BUDGET_H", 11.0)),
        "part": constant(source, "TRAIN_PART", "1/1"),
        "base": constant(source, "TRAIN_BASE", "1/1"),
        "assign": constant(source, "TRAIN_ASSIGN", "").split(),
        "chunk": int(constant(source, "TRAIN_CHUNK", CHUNK)),
        "predecessors": [s.split("/", 1)[1] for s in metadata.get("kernel_sources", [])
                         if not s.endswith("/roadscene-cr-verify")],
    }
    assert config["mode"] == "train", f"{slug} is not a train session"
    return config


def interleave(items, spec):
    index, count = (int(x) for x in spec.split("/"))
    return items[index - 1::count]


def session_part(keys, config, done_before):
    """The key list the session computed at its start (run_kaggle.main_train)."""
    if config["assign"]:
        return [k for k in config["assign"] if k not in done_before]
    remaining = [k for k in interleave(keys, config["base"]) if k not in done_before]
    return interleave(remaining, config["part"])


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def finished_keys(config, part, status):
    """Keys a completed session finished: the first `reports` keys of every chunk it ran."""
    shards = int(status.get("shards", 2))
    jobs = {(j["job"], j["shard"]): j for j in status.get("jobs", []) if str(j.get("job", "")).startswith("T_c")}
    done, problems = set(), []
    for shard in range(shards):
        for number, chunk in enumerate(chunks(part[shard::shards], config["chunk"])):
            job = jobs.get((f"T_c{number:02d}", shard))
            name = f"{config['slug']} T_c{number:02d} s{shard}"
            if job is None:
                problems.append(f"{name}: no job in status.json")
            elif "skipped" in job:
                continue
            elif "error" in job or any(code != 0 for code in job.get("returncodes", [])):
                problems.append(f"{name}: error / return codes {job.get('returncodes')}")
            elif job.get("listed") != len(chunk):
                problems.append(f"{name}: listed {job.get('listed')} but the reconstruction has {len(chunk)}")
            else:
                reports = int(job.get("reports", 0))
                if reports < len(chunk) and not job.get("stopped_by_deadline"):
                    problems.append(f"{name}: {reports}/{len(chunk)} reports without the deadline (failed images?)")
                done.update(chunk[:reports])
    if status.get("this_part") is not None and status["this_part"] != len(part):
        problems.append(f"{config['slug']}: status this_part {status['this_part']} != reconstruction {len(part)}")
    return done, problems


def reachable_keys(config, part):
    """Keys a running session might still reach before its deadline (whole chunks, generous rate)."""
    limit = math.ceil(config["budget_h"] * MAX_RATE / config["chunk"]) * config["chunk"]
    return {k for shard in range(2) for k in part[shard::2][:limit]}, limit


def plan(keys, slugs, running, status_of):
    """status_of(slug) -> the session's status.json dict (completed sessions only)."""
    configs = {slug: session_config(slug) for slug in slugs}
    done_by, reserved_by, problems, report = {}, {}, [], {}
    for slug in slugs:  # in push order: predecessors first
        config = configs[slug]
        missing = [p for p in config["predecessors"] if p not in done_by]
        if missing:
            problems.append(f"{slug}: predecessors {missing} not complete or not listed before it")
            continue
        before = set().union(*(done_by[p] for p in config["predecessors"])) if config["predecessors"] else set()
        part = session_part(keys, config, before)
        if slug in running:
            reach, limit = reachable_keys(config, part)
            reserved_by[slug] = reach
            report[slug] = {"state": "running", "part": len(part), "reach_per_shard": limit, "reserved": len(reach)}
            continue
        status = status_of(slug)
        if status.get("done_before") is not None and status["done_before"] != len(before):
            problems.append(f"{slug}: status done_before {status['done_before']} != reconstruction {len(before)}")
        done, issues = finished_keys(config, part, status)
        problems += issues
        done_by[slug] = done
        report[slug] = {"state": "complete", "part": len(part), "finished": len(done)}
    done = set().union(*done_by.values()) if done_by else set()
    reserved = set().union(*reserved_by.values()) if reserved_by else set()
    free = [k for k in keys if k not in done and k not in reserved]
    pairs = [(a, b) for a in done_by for b in done_by if a < b and done_by[a] & done_by[b]]
    if pairs:
        report["finished_twice"] = {f"{a}|{b}": len(done_by[a] & done_by[b]) for a, b in pairs}
    report.update({"train_keys": len(keys), "finished": len(done), "reserved_by_running": len(reserved),
                   "free": len(free)})
    return free, report, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", required=True, help="every train session so far, in push order")
    parser.add_argument("--running", nargs="*", default=[], help="sessions that are still queued or running")
    parser.add_argument("--split", type=int, default=2, help="number of new sessions to share the free keys")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    keys = KEYS_FILE.read_text(encoding="utf-8").split()

    def status_of(slug):
        return json.loads((DOWNLOADS / slug / "out" / "status.json").read_text(encoding="utf-8"))

    free, report, problems = plan(keys, args.sessions, set(args.running), status_of)
    print(json.dumps(report, indent=1))
    if problems:
        raise SystemExit("NOT PLANNED, inconsistencies:\n  " + "\n  ".join(problems))
    args.out.mkdir(parents=True, exist_ok=True)
    parts = [free[i::args.split] for i in range(args.split)]
    for number, part in enumerate(parts, start=1):
        (args.out / f"assign_{number}.txt").write_text("".join(f"{k}\n" for k in part), encoding="utf-8")
    report["assign_sizes"] = [len(p) for p in parts]
    (args.out / "plan.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("assignments:", report["assign_sizes"], "->", args.out)


if __name__ == "__main__":
    main()
