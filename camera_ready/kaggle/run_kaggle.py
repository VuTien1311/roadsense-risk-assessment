#!/usr/bin/env python3
"""RoadScene-FailAware camera-ready: the remaining GPU inference, run on Kaggle (GPU T4 x2).

Every job is sharded over the GPUs (one harness process per GPU). Every harness call
uses --resume and --stop-at, so a session never stops in the middle of an image.

  bench  full-system latency on 120 images of the 1k subset, alone on GPU 0
  A      the val_rest images that were not finished locally (part of the 9,500-image
         test split): full system + 9 fusion ablations + corridor_road + 2 detector
         swaps, all computed online on the same network outputs
  C      the clean 1k subset, full system + 5 variants: the same-hardware reference
         for B, and the cross-hardware check against the local RTX 3050 runs
  B      the 1k subset under the 6 in-memory perturbations, full system + 5 variants

Results: /kaggle/working/out/<job>_s<shard>.zip (written as soon as a job ends),
status.json, pip_freeze.txt, nvidia_smi.txt, runner.log.

Inputs: the private dataset roadscene-cr-bundle (code, YOLO weights, images and the
manifest) and the private dataset edge-adas-teachers-code (Depth Anything V2 ViT-L and
HybridNets weights). Every weight and image is checked against the SHA-256 values in
the manifest before anything runs; the run stops if the software versions differ
from the local environment that produced the other results.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

SMOKE = 0  # > 0: environment test; every job is limited to SMOKE images per shard
BUDGET_H = 11.0  # no new image is started after this many hours (Kaggle limit: 12 h)
MODE = "val"  # "val": the remaining val work (bench, A, C, B); "train": BDD100K train split (T) + O
TRAIN_PART = "1/1"  # train mode: this session takes part i of n of the train images not done yet
TRAIN_BASE = "1/1"  # train mode: only part i of n of ALL train keys (beside a session that owns the other part)
TRAIN_ASSIGN = ""  # train mode: if set, exactly these train keys (space-separated; replaces TRAIN_BASE/TRAIN_PART)
RUN_OPERATING = False  # train mode: also run the operating points on the 8,400 val_rest images (O)
TRAIN_CHUNK = 2000  # train mode: images per shard and packed chunk (a crash loses one chunk at most)

T0 = time.time()
DEADLINE = T0 + BUDGET_H * 3600
OUT = Path("/kaggle/working/out")
WORK = Path("/tmp/rs_work")
REPO = Path("/tmp/rs/RoadScene-FailAware")
MANIFEST = "roadscene_cr_manifest.json"
TORCH = ["torch==2.7.1", "torchvision==0.22.1"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
PINS = [
    "ultralytics==8.4.66",
    "ultralytics-thop==2.0.20",
    "numpy==2.4.6",
    "opencv-python==4.13.0.92",
    "opencv-python-headless==4.13.0.92",
    "pillow==12.2.0",
    "timm==1.0.27",
    "efficientnet_pytorch==0.7.1",
    "pretrainedmodels==0.7.4",
    "matplotlib==3.10.9",
    "prefetch_generator==1.0.3",
    "webcolors==25.10.0",
    "safetensors==0.8.0",
    "PyYAML==6.0.3",
]
EXPECTED = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "numpy": "2.4.6",
    "opencv": "4.13.0",
    "pillow": "12.2.0",
    "timm": "1.0.27",
    "ultralytics": "8.4.66",
    "matplotlib": "3.10.9",
}
VARIANTS_A = (
    "B0;B1;B2;B3;no_hybridnets;no_depth;single_region;no_scene_quality;no_gates;"
    "corridor_road;swap:yolo11s-seg.pt;swap:yolo11m-seg.pt"
)
VARIANTS_P = "single_region;no_gates;no_scene_quality;B0;B3"
PERTURBATIONS = ("dark", "bright", "contrast", "blur", "noise", "jpeg")
BENCH_IMAGES = 120

_lock = threading.Lock()
STATUS = {"smoke": SMOKE, "budget_h": BUDGET_H, "started_unix": T0, "jobs": []}


def log(message):
    line = f"[{(time.time() - T0) / 3600:6.2f} h] {message}"
    with _lock:
        print(line, flush=True)
        with open(OUT / "runner.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_status():
    with _lock:
        STATUS["updated_h"] = round((time.time() - T0) / 3600, 3)
        temporary = OUT / "status.json.tmp"
        temporary.write_text(json.dumps(STATUS, indent=1), encoding="utf-8")
        temporary.replace(OUT / "status.json")


def sh(command, **kwargs):
    log("$ " + " ".join(str(c) for c in command))
    return subprocess.run([str(c) for c in command], check=False, **kwargs).returncode


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


ENV_CHECK = r"""
import importlib.util, json, sys
import cv2, matplotlib, numpy, PIL, timm, torch, torchvision, ultralytics
print(json.dumps({
    "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(), "torchvision": torchvision.__version__,
    "numpy": numpy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
    "timm": timm.__version__, "ultralytics": ultralytics.__version__,
    "matplotlib": matplotlib.__version__,
    "xformers_importable": importlib.util.find_spec("xformers") is not None,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}))
"""


def install():
    py = sys.executable
    # Other OpenCV builds share the cv2 package folder; remove them so the pinned build is the one imported.
    sh([py, "-m", "pip", "uninstall", "-y", "-q", "opencv-contrib-python", "opencv-contrib-python-headless"])
    sh([py, "-m", "pip", "install", "-q", "--no-cache-dir", *TORCH, "--index-url", TORCH_INDEX])
    sh([py, "-m", "pip", "install", "-q", "--no-cache-dir", *PINS])
    sh([py, "-m", "pip", "uninstall", "-y", "-q", "xformers"])
    result = subprocess.run([py, "-c", ENV_CHECK], capture_output=True, text=True)
    (OUT / "pip_freeze.txt").write_text(
        subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout, encoding="utf-8"
    )
    (OUT / "nvidia_smi.txt").write_text(
        subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, encoding="utf-8"
    )
    if result.returncode != 0:
        return {"error": result.stderr[-3000:]}, ["environment check failed to import the pipeline packages"]
    env = json.loads(result.stdout.strip().splitlines()[-1])
    problems = [
        f"{name}={env.get(name)} (want {want})"
        for name, want in EXPECTED.items()
        if not str(env.get(name, "")).startswith(want)
    ]
    if env["xformers_importable"]:
        problems.append("xformers is importable (Depth Anything V2 would switch attention kernels)")
    if not env["gpus"]:
        problems.append("no GPU visible")
    return env, problems


def bundle_part(root, name, marker):
    """Find a bundle folder whether or not Kaggle extracted the uploaded archive."""
    for candidate in (root / name, root / name / name):
        if (candidate / marker).exists():
            return candidate
    archive = root / f"{name}.zip"
    if archive.exists():
        target = Path("/tmp/rs_bundle") / name
        if not target.exists():
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
        for candidate in (target, target / name):
            if (candidate / marker).exists():
                return candidate
    hits = sorted(root.rglob(marker))
    if hits:
        return hits[0].parent
    raise SystemExit(f"bundle part missing: {name} (looked for {marker} under {root})")


def locate():
    hits = sorted(glob.glob(f"/kaggle/input/**/{MANIFEST}", recursive=True))
    if not hits:
        raise SystemExit("bundle manifest not found under /kaggle/input")
    root = Path(hits[0]).parent
    manifest = json.loads(Path(hits[0]).read_text(encoding="utf-8"))
    markers = {"code": "test3.py", "yolo": "yolo11x-seg.pt", "images": f"{manifest['standard'][0]}.jpg"}
    parts = {name: bundle_part(root, name, marker) for name, marker in markers.items()}
    log(f"bundle parts: {', '.join(f'{k}={v}' for k, v in parts.items())}")
    weights = {}
    for rel, info in manifest["weights"].items():
        if rel.startswith("yolo"):
            weights[rel] = parts["yolo"] / rel
            continue
        candidates = [Path(p) for p in sorted(glob.glob(f"/kaggle/input/**/{rel}", recursive=True))]
        candidates = [p for p in candidates if p.stat().st_size == info["bytes"]]
        if not candidates:
            raise SystemExit(f"weight not found in the attached datasets: {rel}")
        weights[rel] = candidates[0]
    return manifest, parts, weights


def verify(manifest, parts, weights):
    bad = []
    for rel, info in manifest["weights"].items():
        if sha256(weights[rel]) != info["sha256"]:
            bad.append(rel)
    for key, digest in manifest["images"].items():
        path = parts["images"] / f"{key}.jpg"
        if not path.exists() or sha256(path) != digest:
            bad.append(key)
    return bad


def build_repo(parts, weights):
    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(parts["code"], REPO)
    links = {REPO / rel: target for rel, target in weights.items()}
    for link, target in links.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)


def write_list(name, keys, images):
    path = WORK / "lists" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{images / (key + '.jpg')}\n" for key in keys), encoding="utf-8")
    return path, len(keys)


def pack(job_dir, target, cache_parts):
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        bases = [job_dir / "runs"] + [job_dir / "cache" / part for part in cache_parts]
        for base in bases:
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(job_dir).as_posix())
        if (job_dir / "harness.log").exists():
            handle.write(job_dir / "harness.log", "harness.log")
        for extra in sorted(job_dir.glob("*.jsonl")):
            handle.write(extra, extra.name)
    temporary.replace(target)


def harness_job(shard, gpu, job, image_list, out_name, variants, extra, cache_parts):
    path, listed = image_list
    job_dir = WORK / f"s{shard}" / job
    runs = job_dir / "runs"
    job_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "camera_ready/harness.py", "infer", "--image-list", path,
        "--out", runs / out_name, "--cache", job_dir / "cache", "--no-vis", "--no-depth-cache",
        "--resume", "--stop-at", f"{DEADLINE:.0f}", *extra,
    ]
    if variants:
        command += ["--variants", variants]
    if SMOKE:
        command += ["--limit", SMOKE]
        listed = min(listed, SMOKE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    record = {"job": job, "shard": shard, "gpu": gpu, "listed": listed, "returncodes": [],
              "started_h": round((time.time() - T0) / 3600, 3)}
    log(f"{job} s{shard}: start on GPU {gpu}, {listed} images")
    while len(record["returncodes"]) < 3 and time.time() < DEADLINE:
        with open(job_dir / "harness.log", "a", encoding="utf-8") as handle:
            code = subprocess.run([str(c) for c in command], cwd=REPO, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        record["returncodes"].append(code)
        if code == 0:
            break
        log(f"{job} s{shard}: harness exited with {code}; retrying with --resume in 20 s")
        time.sleep(20)
    record["reports"] = len(list((runs / out_name).glob("upgraded_*.json")))
    record["stopped_by_deadline"] = time.time() >= DEADLINE and record["reports"] < listed
    record["ended_h"] = round((time.time() - T0) / 3600, 3)
    record["yolo_cache_dirs"] = sorted(p.name for p in (job_dir / "cache" / "yolo").glob("*")) \
        if (job_dir / "cache" / "yolo").exists() else []
    pack(job_dir, OUT / f"{job}_s{shard}.zip", cache_parts)
    log(f"{job} s{shard}: {record['reports']}/{listed} images, return codes {record['returncodes']}")
    return record


def shard_worker(shard, gpu, lists):
    jobs = [
        ("A", lists[f"A{shard}"], "val_rest_full", VARIANTS_A, [], ["hybridnets", "yolo"]),
        ("C", lists[f"C{shard}"], "stdkaggle_full", VARIANTS_P, ["--no-yolo-cache"], []),
    ]
    for name in PERTURBATIONS:
        jobs.append((f"B_{name}", lists[f"C{shard}"], f"stdpert_{name}_full", VARIANTS_P,
                     ["--no-yolo-cache", "--perturb", name], []))
    for job in jobs:
        if time.time() >= DEADLINE:
            STATUS["jobs"].append({"job": job[0], "shard": shard, "skipped": "deadline"})
            continue
        try:
            STATUS["jobs"].append(harness_job(shard, gpu, *job))
        except Exception:  # keep going: the other jobs and the packing must still happen
            STATUS["jobs"].append({"job": job[0], "shard": shard, "error": traceback.format_exc()[-3000:]})
            log(f"{job[0]} s{shard}: ERROR\n{traceback.format_exc()}")
        write_status()


# ---------------------------------------------------------------------------------------------
# Train mode: the 70k BDD100K train images (T) and the operating points on val_rest (O).
# ---------------------------------------------------------------------------------------------
VARIANTS_T = ("no_hybridnets;risk:scale=0.8;risk:scale=1.2;risk:scale=1.4;risk:scale=1.6;risk:scale=2.0;"
              "no_gates;B0")
VARIANTS_O = "risk:scale=0.8;risk:scale=1.2;risk:scale=1.4;risk:scale=1.6;risk:scale=2.0"


def safe_name(name):
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


TRAIN_DIRS = ["train_full"] + [f"train_{safe_name(v)}" for v in VARIANTS_T.split(";")]


def write_paths(name, paths):
    path = WORK / "lists" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return path, len(paths)


def previous_done():
    """Train images finished in all variant folders by earlier sessions (attached as kernel sources)."""
    # T_c*_s*.zip: outputs of earlier sessions (kernel sources); *T_c*_s*.zipdata: the same archives
    # copied from the team account's sessions into a dataset (renamed so Kaggle does not unpack them).
    archives = sorted(glob.glob("/kaggle/input/**/T_c*_s*.zip", recursive=True)
                      + glob.glob("/kaggle/input/**/*T_c*_s*.zipdata", recursive=True))
    done = set()
    for archive in archives:
        per_dir = {}
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                bits = name.split("/")
                if len(bits) == 3 and bits[0] == "runs" and bits[2].startswith("upgraded_") and bits[2].endswith(".json"):
                    per_dir.setdefault(bits[1], set()).add(bits[2][len("upgraded_"):-len(".json")])
        if all(d in per_dir for d in TRAIN_DIRS):
            done |= set.intersection(*(per_dir[d] for d in TRAIN_DIRS))
    return done, len(archives)


def train_worker(shard, gpu, keys, images, operating):
    jobs = []
    if operating:
        jobs.append(("O", operating, "valop_full", VARIANTS_O, ["--no-yolo-cache"], []))
    for start in range(0, len(keys), TRAIN_CHUNK):
        name = f"T_c{start // TRAIN_CHUNK:02d}"
        boxes = WORK / f"s{shard}" / name / "boxes.jsonl"
        chunk = write_paths(f"{name}_s{shard}.txt", [images[k] for k in keys[start:start + TRAIN_CHUNK]])
        jobs.append((name, chunk, "train_full", VARIANTS_T, ["--no-yolo-cache", "--yolo-boxes", boxes], ["hybridnets"]))
    for job in jobs:
        if time.time() >= DEADLINE:
            STATUS["jobs"].append({"job": job[0], "shard": shard, "skipped": "deadline"})
            continue
        try:
            STATUS["jobs"].append(harness_job(shard, gpu, *job))
        except Exception:
            STATUS["jobs"].append({"job": job[0], "shard": shard, "error": traceback.format_exc()[-3000:]})
            log(f"{job[0]} s{shard}: ERROR\n{traceback.format_exc()}")
        write_status()


def main_train(env):
    manifest, parts, weights = locate()
    code = sorted(glob.glob("/kaggle/input/**/roadscene-cr-train-bundle/code/test3.py", recursive=True))
    if not code:
        raise SystemExit("roadscene-cr-train-bundle code not found")
    parts["code"] = Path(code[0]).parent
    bad = [rel for rel, info in manifest["weights"].items() if sha256(weights[rel]) != info["sha256"]]
    reports = sorted(glob.glob("/kaggle/input/**/verify_report.json", recursive=True))
    report = json.loads(Path(reports[0]).read_text(encoding="utf-8")) if reports else {}
    STATUS.update({"hash_failures": bad, "verify_report": reports[:1], "verify_passed": report.get("all_checks_passed")})
    write_status()
    if bad or not report.get("all_checks_passed"):
        log(f"NOT RUN: weight hash failures {bad}, verification passed = {report.get('all_checks_passed')}")
        raise SystemExit(3)
    keys = Path(reports[0]).with_name("train_keys.txt").read_text(encoding="utf-8").split()
    base = "/kaggle/input/**/bdd100k-edge-adas-data"
    images = {Path(p).stem: Path(p) for p in glob.glob(f"{base}/images_train_*/**/*.jpg", recursive=True)}
    val_images = {Path(p).stem: Path(p) for p in glob.glob(f"{base}/images_val/val/*.jpg", recursive=True)}
    missing = [k for k in keys if k not in images]
    done, archives = previous_done()
    if TRAIN_ASSIGN:
        # A session on another Kaggle account: exactly the keys the driver assigned to it (keys no
        # other session has done or will reach), in the driver's order.
        assigned = TRAIN_ASSIGN.split()
        known = set(keys)
        unknown = [k for k in assigned if k not in known]
        if unknown or len(set(assigned)) != len(assigned):
            raise SystemExit(f"bad assignment: {len(unknown)} unknown keys, {len(assigned) - len(set(assigned))} repeated")
        part = [k for k in assigned if k not in done]
        STATUS.update({"assigned": len(assigned),
                       "assigned_sha256": hashlib.sha256(" ".join(assigned).encode("utf-8")).hexdigest()})
    else:
        # TRAIN_BASE selects the same interleaved half as an earlier "--part i/n" session with no
        # predecessors, so a session can finish that half while another session works on the other one.
        base_index, base_count = (int(x) for x in TRAIN_BASE.split("/"))
        remaining = [k for k in keys[base_index - 1::base_count] if k not in done]
        index, count = (int(x) for x in TRAIN_PART.split("/"))
        part = remaining[index - 1::count]
    if SMOKE:
        part = part[: 2 * SMOKE]
    STATUS.update({"train_keys": len(keys), "missing_images": len(missing), "previous_archives": archives,
                   "done_before": len(done), "this_part": len(part), "part": TRAIN_PART, "base": TRAIN_BASE})
    write_status()
    log(f"train: {len(keys)} keys, {len(done)} done in {archives} earlier chunks, this part {len(part)}")
    if missing:
        raise SystemExit(f"{len(missing)} train images missing, e.g. {missing[:5]}")
    build_repo(parts, weights)
    shards = min(2, len(env["gpus"]))
    STATUS["shards"] = shards
    val_rest = (REPO / "camera_ready" / "sets" / "val_rest_ids.txt").read_text(encoding="utf-8").split()
    threads = []
    for k in range(shards):
        operating = write_paths(f"O_s{k}.txt", [val_images[v] for v in val_rest[k::shards]]) if RUN_OPERATING else None
        threads.append(threading.Thread(target=train_worker, args=(k, k, part[k::shards], images, operating)))
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(120)
        write_status()
    write_status()
    log("finished")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"start: MODE={MODE}, SMOKE={SMOKE}, budget {BUDGET_H} h")
    env, problems = install()
    STATUS["env"], STATUS["problems"] = env, problems
    write_status()
    if problems:
        log("ENVIRONMENT MISMATCH, nothing was run: " + "; ".join(problems))
        raise SystemExit(2)
    if MODE == "train":
        return main_train(env)
    manifest, parts, weights = locate()
    STATUS["local_env"] = manifest.get("local_env")
    bad = verify(manifest, parts, weights)
    STATUS["hash_failures"] = bad
    write_status()
    if bad:
        log(f"HASH MISMATCH for {len(bad)} files, nothing was run: {bad[:10]}")
        raise SystemExit(3)
    log(f"verified {len(manifest['weights'])} weights and {len(manifest['images'])} images")
    build_repo(parts, weights)
    shards = min(2, len(env["gpus"]))
    lists = {}
    a_keys, c_keys = manifest["val_rest_remaining"], manifest["standard"]
    for k in range(shards):
        lists[f"A{k}"] = write_list(f"A_s{k}.txt", a_keys[k::shards], parts["images"])
        lists[f"C{k}"] = write_list(f"C_s{k}.txt", c_keys[k::shards], parts["images"])
    bench = write_list("bench.txt", c_keys[:BENCH_IMAGES], parts["images"])
    STATUS["shards"] = shards
    STATUS["jobs"].append(harness_job(0, 0, "bench", bench, "t4bench_full", "", [], []))
    write_status()
    threads = [threading.Thread(target=shard_worker, args=(k, k, lists)) for k in range(shards)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(120)
        write_status()
    write_status()
    log("finished")


if __name__ == "__main__":
    main()
