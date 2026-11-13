"""Build the private Kaggle dataset bundle for the GPU work left after the local pause.

Writes ``<workspace>/tmp/kaggle_bundle/roadscene-cr-bundle/`` with

  code/     the code the harness imports, plus the dev-split priors (no weights)
  yolo/     the three YOLO11-seg checkpoints
  images/   the val_rest images not finished locally, plus the 1,000-image subset
  roadscene_cr_manifest.json   image keys, SHA-256 of every weight and image,
                               the local software versions and the git commit
  dataset-metadata.json        the private dataset <owner>/roadscene-cr-bundle

Depth Anything V2 and HybridNets weights are not copied: the kernel takes them
from the private dataset edge-adas-teachers-code and checks their SHA-256.

Usage: python camera_ready/kaggle/build_bundle.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "camera_ready"
RUNS = OUT / "runs"
BUNDLE = ROOT.parent / "tmp" / "kaggle_bundle" / "roadscene-cr-bundle"
DATASET_ID = os.environ.get("KAGGLE_OWNER", "your-kaggle-user") + "/roadscene-cr-bundle"
VAL_REST_DIRS = (
    "full", "B0", "B1", "B2", "B3", "no_hybridnets", "no_depth", "single_region",
    "no_scene_quality", "no_gates", "corridor_road", "swap_yolo11s", "swap_yolo11m",
)
WEIGHTS = (
    "checkpoints/depth_anything_v2_vitl.pth",
    "hybrid2/weights/hybridnets.pth",
    "yolo11x-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
)
CODE_FILES = ("test3.py", "paper_visualization.py", "paper_config.json")
CODE_DIRS = ("depth_anything_v2", "hybrid2", "camera_ready")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "weights", "kaggle", ".git")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_list(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finished(directory):
    return {p.stem.replace("upgraded_", "", 1) for p in directory.glob("upgraded_*.json")}


def local_env():
    import cv2
    import matplotlib
    import numpy
    import PIL
    import timm
    import torch
    import torchvision
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "timm": timm.__version__,
        "ultralytics": ultralytics.__version__,
        "matplotlib": matplotlib.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main():
    val_rest = read_list(OUT / "sets" / "val_rest.txt")
    standard = read_list(OUT / "sets" / "standard.txt")
    done = set.intersection(*(finished(RUNS / f"val_rest_{name}") for name in VAL_REST_DIRS))
    remaining = [path for path in val_rest if Path(path).stem not in done]
    assert set(Path(p).stem for p in val_rest[: len(done)]) == done, "finished keys are not a prefix"
    print(f"val_rest: {len(done)} finished locally, {len(remaining)} remaining; subset: {len(standard)}")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "images").mkdir(parents=True)
    (BUNDLE / "yolo").mkdir()
    code = BUNDLE / "code"
    code.mkdir()
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    for name in CODE_DIRS:
        shutil.copytree(ROOT / name, code / name, ignore=IGNORE)
    priors = code / "outputs" / "camera_ready" / "priors"
    priors.mkdir(parents=True)
    for path in sorted((OUT / "priors").glob("*.npy")):
        shutil.copy2(path, priors / path.name)

    images = {}
    for path in remaining + standard:
        key = Path(path).stem
        shutil.copy2(path, BUNDLE / "images" / f"{key}.jpg")
        images[key] = sha256(path)
    weights = {}
    for rel in WEIGHTS:
        source = ROOT / rel
        weights[rel] = {"bytes": source.stat().st_size, "sha256": sha256(source)}
        if rel.startswith("yolo"):
            shutil.copy2(source, BUNDLE / "yolo" / rel)
    declared = json.loads((ROOT / "weights_manifest.json").read_text(encoding="utf-8"))
    for entry in declared["models"] + declared["camera_ready_detector_swap"]:
        assert weights[entry["path"]]["sha256"] == entry["sha256"], f"{entry['path']} differs from weights_manifest.json"

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "camera_ready", "test3.py"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    manifest = {
        "purpose": "RoadScene-FailAware camera-ready: GPU work remaining after the local pause on 2026-11-09",
        "git_commit": commit,
        "code_differs_from_commit": dirty,
        "val_rest_finished_locally": len(done),
        "val_rest_remaining": [Path(p).stem for p in remaining],
        "standard": [Path(p).stem for p in standard],
        "expected_yolo_tag": "yolo11x-seg-56f78e29",
        "images": images,
        "weights": weights,
        "local_env": local_env(),
    }
    (BUNDLE / "roadscene_cr_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (BUNDLE / "dataset-metadata.json").write_text(
        json.dumps({"title": "RoadScene CR bundle", "id": DATASET_ID, "licenses": [{"name": "unknown"}]}, indent=1),
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file()) / 2**20
    print(f"bundle: {BUNDLE} ({size:.0f} MB, {len(images)} images, {len(weights)} weights hashed)")


if __name__ == "__main__":
    main()
