"""Drive the BDD100K train-split phase on Kaggle, one decision per call (safe to rerun).

    python camera_ready/kaggle/train_driver.py            act (push / download / import)
    python camera_ready/kaggle/train_driver.py --dry-run  only report what it would do

Two Kaggle accounts. The main account (the main account, ~/.kaggle) owns the datasets, the
verification kernel, the sessions roadscene-cr-train-<n> and the evaluation. The team account
(<team-account>; its token file is used through KAGGLE_API_TOKEN and the login is checked before every
call, because the CLI silently falls back to the main account) reads the four datasets (shared
read-only) and runs sessions roadscene-cr-team-<n> on explicitly assigned keys.

State: outputs/camera_ready/kaggle/train_state.json ({"sessions": [...], "team_sessions": [...],
"pending": [...], "operating_session": ..., "team_out": ..., "eval_kernel": ..., "imported": false}).
Order of decisions:

0. Pre-planned main sessions ("pending", folders made with make_kernel.py) are pushed as soon as
   a GPU slot is free and every earlier session they read (kernel_sources) is complete.
1. The val_rest operating points (job O) are imported as soon as their session is complete.
2. A kernel status that is unknown, error or cancelled stops the driver (checked by hand).
3. Free train keys = keys no complete session finished and no running session can reach
   (team_plan.py, reconstructed exactly from every session's settings and status.json). While
   a main session runs, free keys wait (its final progress decides the gaps). Then they are
   pushed as new sessions with explicit key lists, team account first, on free GPU slots
   (Kaggle allows 2 per account) and only with at least MIN_QUOTA_H hours of GPU quota left.
4. When no key is free and nothing runs: the team sessions' archives are downloaded, checked
   against the plan, renamed <session>__T_c*_s*.zipdata (so Kaggle keeps them packed) and
   uploaded as the main account's private dataset roadscene-cr-train-team-out; once it is ready
   and complete, the CPU evaluation kernel (eval_train.py) is pushed with it attached.
5. When the evaluation is complete: download eval_train.zip into outputs/camera_ready/eval/,
   check that every train key is complete in every run, and write train_provenance.json.

Never runs GPU work locally, never overwrites existing run or evaluation folders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import team_plan  # noqa: E402

ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "outputs" / "camera_ready"
STATE = OUT / "kaggle" / "train_state.json"
KAGGLE = WORKSPACE / "tmp" / "kaggle_cli" / "Scripts" / "kaggle.exe"
DOWNLOADS = WORKSPACE / "tmp" / "kaggle_out"
KERNELS = WORKSPACE / "tmp" / "kaggle_kernels"
PLANS = WORKSPACE / "tmp" / "team_plan" / "assigned"
TEAM_OUT_DIR = WORKSPACE / "tmp" / "team_out_dataset"
OWNER = os.environ.get("KAGGLE_OWNER", "your-kaggle-user")
TEAM = os.environ.get("KAGGLE_TEAM_OWNER", "second-kaggle-user")
TEAM_TOKEN = Path(os.environ.get("KAGGLE_TEAM_TOKEN", str(Path.home() / ".kaggle_team" / "access_token")))
TEAM_WHOAMI = WORKSPACE / "tmp" / "kaggle_team_whoami.py"
TEAM_INPUT = f"{TEAM}/roadscene-cr-team-input"
TEAM_OUT = f"{OWNER}/roadscene-cr-train-team-out"
MIN_QUOTA_H = 3.0
MAX_GPU_SESSIONS = 2  # per account. Kaggle: "Maximum batch GPU session count of 2"
VALOP_DIRS = ["valop_full"] + [f"valop_risk_scale_{s}" for s in ("0.8", "1.2", "1.4", "1.6", "2.0")]
TRAIN_DIRS = ["train_full", "train_no_hybridnets", "train_risk_scale_0.8", "train_risk_scale_1.2",
              "train_risk_scale_1.4", "train_risk_scale_1.6", "train_risk_scale_2.0", "train_no_gates", "train_B0"]
_TEAM_CHECKED = []


def account_env(owner):
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR")}
    if owner == TEAM:
        env["KAGGLE_API_TOKEN"], env["KAGGLE_CONFIG_DIR"] = str(TEAM_TOKEN), str(TEAM_TOKEN.parent)
        if not _TEAM_CHECKED:
            check = subprocess.run([str(KAGGLE.parent / "python.exe"), str(TEAM_WHOAMI)], env=env,
                                   capture_output=True, text=True)
            if check.returncode != 0 or not check.stdout.startswith(f"user={TEAM} "):
                raise SystemExit(f"team login check failed, nothing run: {check.stdout.strip()} {check.stderr.strip()[-300:]}")
            _TEAM_CHECKED.append(True)
    elif owner != OWNER:
        raise ValueError(owner)
    return env


def kaggle(*args, check=False, owner=OWNER):
    result = subprocess.run([str(KAGGLE), *map(str, args)], capture_output=True, text=True, cwd=WORKSPACE / "tmp",
                            env=account_env(owner))
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(map(str, args))} failed: {result.stdout[-500:]} {result.stderr[-500:]}")
    return result.stdout + result.stderr


def owner_of(state, slug):
    return TEAM if slug in state.get("team_sessions", []) else OWNER


def status(slug, owner=OWNER, attempts=3):
    """queued / running / complete / error / cancel, read from the CLI's status line only. An API
    failure (its message may contain the word "error") is retried and then reported as unknown,
    never as a failed kernel."""
    for attempt in range(attempts):
        text = kaggle("kernels", "status", f"{owner}/{slug}", owner=owner)
        match = re.search(r'has status "KernelWorkerStatus\.([A-Z_]+)"', text)
        if match:
            word = match.group(1).lower()
            return next((known for known in ("queued", "running", "complete", "error", "cancel") if known in word), word)
        if attempt + 1 < attempts:
            time.sleep(20)
    return "unknown"


def quota_gpu_hours(owner=OWNER):
    for row in csv.reader(io.StringIO(kaggle("quota", "-v", owner=owner))):
        if row and row[0].strip().upper() == "GPU":
            return float(row[2].strip().rstrip("h")), row[4].strip()
    return 0.0, "?"


def download(slug, pattern, owner=OWNER):
    target = DOWNLOADS / slug
    target.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{owner}/{slug}", "-p", str(target).replace("/", "\\"), "--file-pattern", pattern, "-o",
           owner=owner)
    return target


def load_state():
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"sessions": [], "pending": [], "operating_session": None, "eval_kernel": None, "imported": False}
    state.setdefault("team_sessions", [])
    state.setdefault("pending", [])
    return state


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def train_keys():
    report = DOWNLOADS / "roadscene-cr-verify"  # the verification output never changes: download once
    if not ((report / "verify_report.json").exists() and (report / "train_keys.txt").exists()):
        report = download("roadscene-cr-verify", r"(verify_report\.json|train_keys\.txt)$")
    verified = json.loads((report / "verify_report.json").read_text(encoding="utf-8"))
    return verified, (report / "train_keys.txt").read_text(encoding="utf-8").split()


def session_status(state, slug):
    """status.json of a complete session (never changes afterwards: downloaded once)."""
    path = DOWNLOADS / slug / "out" / "status.json"
    if not path.exists():
        download(slug, r"out/status\.json$", owner_of(state, slug))
    return json.loads(path.read_text(encoding="utf-8"))


def free_keys(state, keys, running):
    sessions = state["sessions"] + state["team_sessions"]
    return team_plan.plan(keys, sessions, running, lambda slug: session_status(state, slug))


def next_slug(state, owner):
    if owner == TEAM:
        used = [int(s.rsplit("-", 1)[1]) for s in state["team_sessions"]]
        return f"roadscene-cr-team-{max(used, default=0) + 1}"
    used = [int(s.rsplit("-", 1)[1]) for s in state["sessions"] + state.get("pending", []) if s.rsplit("-", 1)[1].isdigit()]
    return f"roadscene-cr-train-{max(used, default=0) + 1}"


def push_free_sessions(state, dry_run, free, running, states):
    """New sessions with explicit key lists for the free keys: team account first, then the main
    account; two per account in parallel when two GPU slots are free and there are >= 4,000 keys."""
    for owner in (TEAM, OWNER):
        own = state["team_sessions"] if owner == TEAM else ["roadscene-cr-run"] + state["sessions"]
        busy = sum(1 for slug in own if slug in running or states.get(slug) in ("queued", "running"))
        slots = MAX_GPU_SESSIONS - busy
        if slots <= 0:
            print(f"{owner}: no free GPU slot ({busy} sessions running)")
            continue
        hours, refresh = quota_gpu_hours(owner)
        if hours < MIN_QUOTA_H:
            print(f"{owner}: only {hours:.1f} GPU hours left; quota reset at {refresh}")
            continue
        count = min(slots, 2 if len(free) >= 4000 else 1)
        budget = round(min(11.0, (hours - 0.5) / count), 2)
        for index in range(count):
            part = free[index::count]
            slug = next_slug(state, owner)
            assign = PLANS / f"{slug}.txt"
            command = [sys.executable, str(HERE / "make_kernel.py"), "--slug", slug, "--mode", "train",
                       "--budget-h", str(budget), "--assign", str(assign)]
            if owner == TEAM:
                command += ["--owner", TEAM, "--input-dataset", TEAM_INPUT]
            print(f"push {owner} GPU session {slug}: {len(part)} assigned keys, budget {budget} h")
            if dry_run:
                continue
            assign.parent.mkdir(parents=True, exist_ok=True)
            assign.write_text("".join(f"{k}\n" for k in part), encoding="utf-8")
            subprocess.run(command, check=True)
            print(kaggle("kernels", "push", "-p", str(KERNELS / slug).replace("/", "\\"), "--accelerator", "NvidiaTeslaT4",
                         check=True, owner=owner))
            (state["team_sessions"] if owner == TEAM else state["sessions"]).append(slug)
            save_state(state)
        return
    print("no account can take the free keys now")


def pending_dependencies(slug):
    """Earlier train sessions a pre-planned kernel reads (its kernel_sources): they must be complete."""
    metadata = json.loads((KERNELS / slug / "kernel-metadata.json").read_text(encoding="utf-8"))
    return [s.split("/", 1)[1] for s in metadata.get("kernel_sources", []) if s != f"{OWNER}/roadscene-cr-verify"]


def archive_keys(path):
    """Train keys with a report in every run folder of one session archive."""
    per_dir = {}
    with zipfile.ZipFile(path) as handle:
        for name in handle.namelist():
            bits = name.split("/")
            if len(bits) == 3 and bits[0] == "runs" and bits[2].startswith("upgraded_") and bits[2].endswith(".json"):
                per_dir.setdefault(bits[1], set()).add(bits[2][len("upgraded_"):-len(".json")])
    return set.intersection(*(per_dir.get(d, set()) for d in TRAIN_DIRS))


def team_out_ready(state, dry_run, keys):
    """Copy the team sessions' archives into the main account's private dataset TEAM_OUT; True once
    the dataset holds exactly the current archives and Kaggle has finished processing it."""
    sessions = list(state["team_sessions"])
    info = state.get("team_out") or {}
    if info.get("sessions") == sessions:
        text = kaggle("datasets", "status", TEAM_OUT).strip()
        if "ready" not in text.lower():
            print("team archive dataset not ready yet:", text[-200:])
            return False
        if not info.get("listed"):
            listing = kaggle("datasets", "files", "-d", TEAM_OUT, "--page-size", "200", "-v")
            names = {row[0] for row in csv.reader(io.StringIO(listing)) if row}
            missing = sorted(set(info["files"]) - names)
            if missing:
                print("team archive dataset lacks", len(missing), "files (unpacked by Kaggle?), e.g.", missing[:3])
                return False
            info["listed"] = True
            save_state(state)
        return True
    print("copy the team sessions' archives into", TEAM_OUT, "for", sessions)
    if dry_run:
        return False
    _, report, problems = free_keys(state, keys, set())
    if problems:
        raise SystemExit("plan inconsistencies, nothing uploaded:\n  " + "\n  ".join(problems))
    planned = {slug: team_plan_finished(state, keys, slug) for slug in sessions}
    staging = TEAM_OUT_DIR / time.strftime("%Y%m%d-%H%M%S")  # a fresh copy per upload; nothing is deleted
    staging.mkdir(parents=True)
    files, hashes, found = [], {}, {}
    for slug in sessions:
        folder = download(slug, r"out/T_c\d+_s\d\.zip$", TEAM)
        archives = sorted((folder / "out").glob("T_c*_s*.zip"))
        found[slug] = set().union(*(archive_keys(a) for a in archives)) if archives else set()
        if found[slug] != planned[slug]:
            raise SystemExit(f"{slug}: archives hold {len(found[slug])} finished keys, the plan {len(planned[slug])}; "
                             "nothing uploaded")
        for archive in archives:
            name = f"{slug}__{archive.stem}.zipdata"
            shutil.copyfile(archive, staging / name)
            hashes[name] = hashlib.sha256((staging / name).read_bytes()).hexdigest()
            files.append(name)
    union = set().union(*found.values()) if found else set()
    print(f"team archives: {len(files)} files, {len(union)} finished keys")
    metadata = {"title": "RoadScene CR train team out", "id": TEAM_OUT, "licenses": [{"name": "unknown"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    exists = "ready" in kaggle("datasets", "status", TEAM_OUT).lower()
    if exists:
        print(kaggle("datasets", "version", "-p", str(staging).replace("/", "\\"), "-m",
                     f"team sessions {', '.join(sessions)}", check=True)[-300:])
    else:
        print(kaggle("datasets", "create", "-p", str(staging).replace("/", "\\"), check=True)[-300:])
    state["team_out"] = {"sessions": sessions, "files": files, "sha256": hashes, "finished_keys": len(union),
                         "uploaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(state)
    return False


def team_plan_finished(state, keys, slug):
    """Keys the plan says a complete session finished (prefix of every chunk, from status.json)."""
    config = team_plan.session_config(slug)
    part = team_plan.session_part(keys, config, set())
    done, problems = team_plan.finished_keys(config, part, session_status(state, slug))
    if problems:
        raise SystemExit(f"{slug}: " + "; ".join(problems))
    return done


def push_eval(state, dry_run):
    slug = "roadscene-cr-train-eval"
    folder = KERNELS / slug
    print("push CPU evaluation kernel", slug)
    if dry_run:
        return
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "eval_train.py").write_bytes((HERE / "eval_train.py").read_bytes())
    metadata = {
        "id": f"{OWNER}/{slug}", "title": "Roadscene CR Train Eval", "code_file": "eval_train.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{OWNER}/roadscene-cr-train-bundle",
                            "nguyentrongquocdat/object-detection-and-drivable-lane-masking"]
                           + ([TEAM_OUT] if state["team_sessions"] else []),
        "competition_sources": [], "model_sources": [],
        "kernel_sources": [f"{OWNER}/roadscene-cr-verify"] + [f"{OWNER}/{s}" for s in state["sessions"]],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(folder).replace("/", "\\"), check=True))
    state["eval_kernel"] = slug
    save_state(state)


def import_operating(state, dry_run):
    """The val_rest operating points (job O) do not depend on the train evaluation: import them as
    soon as their session is complete."""
    print("import the val_rest operating points from", state["operating_session"])
    if dry_run:
        return
    runs, eval_dir = OUT / "runs", OUT / "eval"
    op = download(state["operating_session"], r"out/O_s\d\.zip$")
    archives = sorted(op.rglob("O_s*.zip"))
    if not archives:
        print("no O_s*.zip in", state["operating_session"])
        return
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                bits = member.split("/")
                if len(bits) == 3 and bits[0] == "runs" and bits[1] in VALOP_DIRS and not (runs / bits[1] / bits[2]).exists():
                    (runs / bits[1]).mkdir(parents=True, exist_ok=True)
                    (runs / bits[1] / bits[2]).write_bytes(handle.read(member))
    for name in VALOP_DIRS:
        if (runs / name).exists() and not (eval_dir / name / "images.csv").exists():
            subprocess.run([sys.executable, str(ROOT / "camera_ready" / "gt_eval.py"), "--run", str(runs / name),
                            "--split", "val_rest", "--name", name], check=True)
    state["operating_imported"] = True
    save_state(state)


def import_results(state, verified, keys, dry_run):
    print("import train evaluation")
    if dry_run:
        return
    folder = download(state["eval_kernel"], r"(eval_train\.zip|eval_status\.json)$")
    eval_status = json.loads(next(folder.rglob("eval_status.json")).read_text(encoding="utf-8"))
    complete = eval_status.get("complete_in_all_runs")
    returncodes = eval_status.get("gt_eval_returncodes") or {}
    failed = {run: returncodes.get(run) for run in TRAIN_DIRS if returncodes.get(run) != 0}
    archive = next(folder.rglob("eval_train.zip"))
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        rows = {}
        for run in TRAIN_DIRS:
            if f"eval/{run}/images.csv" in names and f"eval/{run}/objects.csv" in names:
                with handle.open(f"eval/{run}/images.csv") as raw:
                    rows[run] = sum(1 for _ in csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))) - 1
    short = {run: rows.get(run) for run in TRAIN_DIRS if rows.get(run) != len(keys)}
    if (complete != len(keys) or eval_status.get("train_keys") != len(keys) or failed or short
            or eval_status.get("duplicates") != 0):
        # Every train key must be finished and evaluated in every run, from exactly one archive (the
        # sessions partition the keys), before the numbers go into the paper.
        state["eval_incomplete"] = {"complete_in_all_runs": complete, "train_keys": len(keys),
                                    "per_run": eval_status.get("per_run"), "failed_runs": failed,
                                    "image_rows_not_matching": short, "duplicates": eval_status.get("duplicates")}
        save_state(state)
        print("NOT IMPORTED:", json.dumps(state["eval_incomplete"])[:800])
        return
    eval_dir = OUT / "eval"
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            if member.startswith("eval/train_") and not (OUT / member).exists():
                handle.extract(member, OUT)
            elif member == "sets/train_ids.txt":  # published with the other split ID files
                (ROOT / "camera_ready" / "sets" / "train_ids.txt").write_bytes(handle.read(member).replace(b"\r\n", b"\n"))
    if state.get("operating_session") and not state.get("operating_imported"):
        import_operating(state, dry_run)
    provenance = {"sessions": state["sessions"], "team_sessions": state["team_sessions"],
                  "team_account": TEAM if state["team_sessions"] else None, "team_out": state.get("team_out"),
                  "operating_session": state.get("operating_session"),
                  "eval_kernel": state["eval_kernel"], "eval_status": eval_status,
                  "verify": {"all_checks_passed": verified.get("all_checks_passed"), "checks": verified.get("checks"),
                             "train": verified.get("train")}}
    (OUT / "kaggle" / "train_provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    state["imported"] = True
    state.pop("eval_incomplete", None)
    save_state(state)
    print("imported:", json.dumps(eval_status)[:400])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state = load_state()
    if state.get("imported"):
        print("train phase already imported")
        return
    if status("roadscene-cr-verify") != "complete":
        print("verification kernel not complete:", status("roadscene-cr-verify"))
        return
    verified, keys = train_keys()
    if not verified.get("all_checks_passed"):
        print("verification failed; nothing is pushed. Report:", json.dumps(verified.get("checks"))[:600])
        return
    sessions = state["sessions"] + state["team_sessions"]
    kernels = sessions + ([state["eval_kernel"]] if state.get("eval_kernel") else [])
    states = {slug: status(slug, owner_of(state, slug)) for slug in kernels}
    print("kernels:", states)
    # Kaggle runs at most 2 batch GPU sessions at a time; pre-planned sessions wait for a free slot.
    busy = sum(1 for slug in ["roadscene-cr-run"] + state["sessions"]
               if (states.get(slug) or status(slug)) in ("queued", "running"))
    while state.get("pending") and busy < MAX_GPU_SESSIONS:
        slug = state["pending"][0]
        deps = {dep: states.get(dep) or status(dep) for dep in pending_dependencies(slug)}
        if any(s in ("error", "cancel") for s in deps.values()):
            print("drop pre-planned session", slug, "- an earlier session it reads failed:", deps)
            if args.dry_run:
                break
            state["pending"].pop(0)
            save_state(state)
            continue
        if any(s != "complete" for s in deps.values()):
            print("pre-planned session", slug, "waits for", deps)
            break
        print("push pre-planned GPU session", slug)
        if args.dry_run:
            break
        print(kaggle("kernels", "push", "-p", str(KERNELS / slug).replace("/", "\\"), "--accelerator", "NvidiaTeslaT4", check=True))
        state["sessions"].append(state["pending"].pop(0))
        save_state(state)
        states[slug], busy = "running", busy + 1
    op_session = state.get("operating_session")
    if op_session and not state.get("operating_imported") and states.get(op_session) == "complete":
        import_operating(state, args.dry_run)
    odd = {slug: s for slug, s in states.items() if s in ("unknown", "error", "cancel")}
    if odd:
        # An API error must never look like "not running" (that could push duplicate sessions), and a
        # failed session needs a look by hand before its keys are given to anyone else.
        print("a kernel status is unknown, error or cancelled; nothing to do now:", odd)
        return
    if state.get("eval_kernel"):
        eval_state = states.get(state["eval_kernel"])
        if eval_state != "complete":
            print("evaluation kernel status:", eval_state)
            return
        import_results(state, verified, keys, args.dry_run)
        return
    running = {slug for slug in sessions if states.get(slug) in ("queued", "running")}
    free, report, problems = free_keys(state, keys, running)
    print("plan:", json.dumps({k: v for k, v in report.items() if not k.startswith("roadscene")}))
    if problems:
        print("plan inconsistencies; nothing pushed:\n  " + "\n  ".join(problems))
        return
    if free:
        main_running = [slug for slug in state["sessions"] if slug in running]
        if main_running:
            print(f"{len(free)} free train keys wait until {main_running} end (their final progress decides the gaps)")
            return
        push_free_sessions(state, args.dry_run, free, running, states)
        return
    if running:
        print("no free train keys; sessions still running:", sorted(running))
        return
    if state["team_sessions"] and not team_out_ready(state, args.dry_run, keys):
        return
    push_eval(state, args.dry_run)


if __name__ == "__main__":
    main()
