"""Build the private Kaggle dataset for the 70k-image BDD100K train-split evaluation.

Writes ``<workspace>/tmp/kaggle_bundle/roadscene-cr-train-bundle/`` with

  code/        the current code (harness --yolo-boxes, gt_eval --labels/--yolo-boxes, ...)
               plus the dev-split priors; no weights (they come from roadscene-cr-bundle
               and edge-adas-teachers-code and are hash-checked by the kernels)
  reference/   what the Kaggle kernels compare a third-party BDD100K copy against:
               val_labels.json    every val box, class, occluded/truncated flag and image
                                  attribute of the SHA-checked ground truth used for the
                                  headline results (HF mirror dgural/bdd100k)
               val_drivable.json  MD5 of every decoded val drivable mask
               val_images.json    SHA-256 of the 10,000 local val images
               train_images.json  SHA-256 of the local train images (a small subset)
  dataset-metadata.json  private dataset <owner>/roadscene-cr-train-bundle

Usage: python camera_ready/kaggle/build_train_bundle.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import os
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_bundle import CODE_DIRS, CODE_FILES, IGNORE, OUT, ROOT, sha256  # noqa: E402

DATA = ROOT.parent / "data"
LABELS = DATA / "bdd100k_labels_val"
BUNDLE = ROOT.parent / "tmp" / "kaggle_bundle" / "roadscene-cr-train-bundle"
DATASET_ID = os.environ.get("KAGGLE_OWNER", "your-kaggle-user") + "/roadscene-cr-train-bundle"


def val_reference():
    payload = json.loads((LABELS / "samples.json").read_text(encoding="utf-8"))
    labels, drivable = {}, {}
    for sample in payload["samples"]:
        key = Path(sample["filepath"]).stem
        width, height = sample["metadata"]["width"], sample["metadata"]["height"]
        objects = []
        for det in (sample.get("detections") or {}).get("detections", []) or []:
            x, y, w, h = det["bounding_box"]
            objects.append([det["label"], x * width, y * height, (x + w) * width, (y + h) * height,
                            bool(det.get("occluded", False)), bool(det.get("truncated", False))])
        labels[key] = {
            "size": [width, height],
            "attributes": {field: (sample.get(field) or {}).get("label") or "undefined"
                           for field in ("weather", "timeofday", "scene")},
            "objects": objects,
        }
        mask = cv2.imread(str(LABELS / sample["drivable"]["mask_path"]), cv2.IMREAD_UNCHANGED)
        drivable[key] = hashlib.md5(mask.tobytes()).hexdigest() + f":{mask.shape[0]}x{mask.shape[1]}"
    return labels, drivable


def main():
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    code = BUNDLE / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    for name in CODE_DIRS:
        shutil.copytree(ROOT / name, code / name, ignore=IGNORE)
    shutil.copytree(HERE, code / "camera_ready" / "kaggle", ignore=shutil.ignore_patterns("__pycache__"))
    priors = code / "outputs" / "camera_ready" / "priors"
    priors.mkdir(parents=True)
    for path in sorted((OUT / "priors").glob("*.npy")):
        shutil.copy2(path, priors / path.name)

    reference = BUNDLE / "reference"
    reference.mkdir()
    labels, drivable = val_reference()
    (reference / "val_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (reference / "val_drivable.json").write_text(json.dumps(drivable), encoding="utf-8")
    val_images = {p.stem: sha256(p) for p in sorted((DATA / "images" / "100k" / "val").glob("*.jpg"))}
    train_images = {p.stem: sha256(p) for p in sorted((DATA / "images" / "100k" / "train").glob("*.jpg"))}
    (reference / "val_images.json").write_text(json.dumps(val_images), encoding="utf-8")
    (reference / "train_images.json").write_text(json.dumps(train_images), encoding="utf-8")
    (BUNDLE / "dataset-metadata.json").write_text(
        json.dumps({"title": "RoadScene CR train bundle", "id": DATASET_ID, "licenses": [{"name": "unknown"}]}, indent=1),
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file()) / 2**20
    print(f"bundle: {BUNDLE} ({size:.0f} MB): val labels {len(labels)}, val masks {len(drivable)}, "
          f"val images {len(val_images)}, train images {len(train_images)}")


if __name__ == "__main__":
    main()
