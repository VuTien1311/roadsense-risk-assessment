#!/usr/bin/env python3
"""Verify third-party BDD100K copies against the SHA-checked val ground truth, then build the
train ground truth in the gt_eval format (Kaggle CPU kernel).

The official BDD100K download server no longer answers, so the train labels and images come
from Kaggle copies. They are trusted only if, on the val split, they reproduce the ground truth
used for the headline results exactly:

  labels   det_20 det_val.json, converted by the same code as the train labels, must give the
           same objects (class, box within 0.01 px, occluded, truncated) and the same image
           attributes (weather, time of day, scene) as the reference for every val image;
           the det_train.json copies must be byte-identical to each other
  masks    every decoded val drivable mask must be pixel-identical to the reference
  images   val images must match the local SHA-256 values; the local train images (a subset)
           must match too; the train images of two independent copies must be identical

Outputs (/kaggle/working): verify_report.json, train_keys.txt, train_images.json (key -> path),
train_labels/samples.json (gt_eval format; masks are linked in by the evaluation kernel).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

WORK = Path("/kaggle/working")
ROAD_USERS = {"pedestrian", "rider", "other person", "car", "truck", "bus", "train", "motorcycle",
              "bicycle", "other vehicle", "trailer"}
T0 = time.time()
REPORT = {"checks": {}, "notes": []}


def log(message):
    print(f"[{time.time() - T0:7.0f} s] {message}", flush=True)


def find(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_many(paths):
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(sha256, paths))


def convert(frames, width=1280, height=720):
    """det_20 frames -> {key: {"size", "attributes", "objects"}} (all classes, pixel boxes)."""
    out = {}
    for frame in frames:
        key = Path(frame["name"]).stem
        attributes = frame.get("attributes") or {}
        objects = []
        for label in frame.get("labels") or []:
            box = label.get("box2d")
            if not box:
                continue
            flags = label.get("attributes") or {}
            objects.append([label["category"], box["x1"], box["y1"], box["x2"], box["y2"],
                            bool(flags.get("occluded", False)), bool(flags.get("truncated", False))])
        out[key] = {"size": [width, height],
                    "attributes": {f: attributes.get(f) or "undefined" for f in ("weather", "timeofday", "scene")},
                    "objects": objects}
    return out


def same_record(a, b, tol=0.01):
    if a["attributes"] != b["attributes"] or len(a["objects"]) != len(b["objects"]):
        return False
    left = sorted(a["objects"], key=lambda o: (o[0], round(o[1], 1), round(o[2], 1)))
    right = sorted(b["objects"], key=lambda o: (o[0], round(o[1], 1), round(o[2], 1)))
    for x, y in zip(left, right):
        if x[0] != y[0] or x[5] != y[5] or x[6] != y[6] or max(abs(p - q) for p, q in zip(x[1:5], y[1:5])) > tol:
            return False
    return True


def mask_md5(path):
    array = np.array(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return hashlib.md5(array.astype(np.uint8).tobytes()).hexdigest() + f":{array.shape[0]}x{array.shape[1]}"


def main():
    ref_dir = Path(find("reference/val_labels.json")[0]).parent
    ref_labels = json.loads((ref_dir / "val_labels.json").read_text())
    ref_masks = json.loads((ref_dir / "val_drivable.json").read_text())
    ref_val_images = json.loads((ref_dir / "val_images.json").read_text())
    ref_train_images = json.loads((ref_dir / "train_images.json").read_text())

    # 1. det_20 labels
    train_files = find("det_20/det_train.json") + find("obj_labels/det_train.json")
    val_files = find("det_20/det_val.json")
    digests = {path: sha256(path) for path in train_files}
    REPORT["checks"]["det_train_copies"] = {p: d[:16] for p, d in digests.items()}
    REPORT["checks"]["det_train_copies_identical"] = len(set(digests.values())) == 1
    log(f"det_train copies: {len(digests)}, identical={len(set(digests.values())) == 1}")
    val_frames = json.loads(Path(val_files[0]).read_text())
    converted_val = convert(val_frames)
    matched = [k for k in ref_labels if k in converted_val and same_record(ref_labels[k], converted_val[k])]
    bad = [k for k in ref_labels if k not in matched][:20]
    REPORT["checks"]["val_labels"] = {"reference": len(ref_labels), "det_val_frames": len(val_frames),
                                      "identical": len(matched), "examples_different": bad}
    log(f"val labels identical: {len(matched)}/{len(ref_labels)}")

    # 2. drivable masks
    mask_dirs = {split: sorted({str(Path(p).parent) for p in find(f"drivable_labels/masks/{split}/*.png")[:1]})
                 for split in ("train", "val")}
    val_mask_dir = Path(mask_dirs["val"][0])
    keys = sorted(ref_masks)
    with ThreadPoolExecutor(max_workers=8) as pool:
        md5s = list(pool.map(lambda k: mask_md5(val_mask_dir / f"{k}.png") if (val_mask_dir / f"{k}.png").exists() else None, keys))
    same = sum(1 for k, m in zip(keys, md5s) if m == ref_masks[k])
    REPORT["checks"]["val_masks"] = {"reference": len(keys), "identical": same, "dir": str(val_mask_dir)}
    log(f"val masks identical: {same}/{len(keys)}")
    train_mask_dir = Path(mask_dirs["train"][0])

    # 3. images
    sources = {}
    for name, pattern in (("soumikrakshit", "bdd100k_images_100k/bdd100k/images/100k/train"),
                          ("user", "images_train_*")):
        paths = [p for p in glob.glob(f"/kaggle/input/**/{pattern}/**/*.jpg", recursive=True)]
        sources[name] = {Path(p).stem: p for p in paths}
        log(f"train image source {name}: {len(sources[name])} images")
    val_sources = {"soumikrakshit": find("bdd100k_images_100k/bdd100k/images/100k/val/*.jpg"),
                   "user": find("images_val/val/*.jpg")}
    for name, paths in val_sources.items():
        by_key = {Path(p).stem: p for p in paths}
        keys = [k for k in ref_val_images if k in by_key]
        hashes = hash_many([by_key[k] for k in keys])
        REPORT["checks"][f"val_images_{name}"] = {"reference": len(ref_val_images), "found": len(keys),
                                                  "identical": sum(h == ref_val_images[k] for k, h in zip(keys, hashes))}
        log(f"val images {name}: {REPORT['checks'][f'val_images_{name}']}")
    for name, by_key in sources.items():
        keys = [k for k in ref_train_images if k in by_key]
        hashes = hash_many([by_key[k] for k in keys])
        REPORT["checks"][f"local_train_images_{name}"] = {"reference": len(ref_train_images), "found": len(keys),
                                                          "identical": sum(h == ref_train_images[k] for k, h in zip(keys, hashes))}
        log(f"local train images {name}: {REPORT['checks'][f'local_train_images_{name}']}")
    common = sorted(set(sources["soumikrakshit"]) & set(sources["user"]))
    a = hash_many([sources["soumikrakshit"][k] for k in common])
    b = hash_many([sources["user"][k] for k in common])
    differ = [k for k, x, y in zip(common, a, b) if x != y]
    REPORT["checks"]["train_images_cross_copy"] = {"common": len(common), "identical": len(common) - len(differ),
                                                   "examples_different": differ[:20]}
    log(f"train images identical across the two copies: {len(common) - len(differ)}/{len(common)}")

    # 4. train ground truth in the gt_eval format
    frames = json.loads(Path(train_files[0]).read_text())
    converted = convert(frames)
    keys = [k for k in converted if k in sources["user"] and (train_mask_dir / f"{k}.png").exists()]
    samples = []
    for key in keys:
        record = converted[key]
        width, height = record["size"]
        detections = [{"label": o[0], "bounding_box": [o[1] / width, o[2] / height, (o[3] - o[1]) / width, (o[4] - o[2]) / height],
                       "occluded": o[5], "truncated": o[6]} for o in record["objects"] if o[0] in ROAD_USERS]
        samples.append({"filepath": f"{key}.jpg", "metadata": {"width": width, "height": height},
                        "detections": {"detections": detections},
                        **{f: {"label": record["attributes"][f]} for f in ("weather", "timeofday", "scene")},
                        "drivable": {"mask_path": f"fields/drivable/{key}.png"}})
    (WORK / "train_labels").mkdir(parents=True, exist_ok=True)
    (WORK / "train_labels" / "samples.json").write_text(json.dumps({"samples": samples}))
    (WORK / "train_keys.txt").write_text("".join(f"{k}\n" for k in keys))
    (WORK / "train_images.json").write_text(json.dumps({k: sources["user"][k] for k in keys}))
    import cv2  # gt_eval reads masks with cv2; it must see the same values as PIL

    values, modes, cv2_same = set(), set(), 0
    sample_keys = keys[:: max(1, len(keys) // 300)]
    for key in sample_keys:
        image = Image.open(train_mask_dir / f"{key}.png")
        modes.add(image.mode)
        array = np.array(image)
        values |= set(np.unique(array).tolist())
        other = cv2.imread(str(train_mask_dir / f"{key}.png"), cv2.IMREAD_UNCHANGED)
        other = other[..., 0] if other is not None and other.ndim == 3 else other
        cv2_same += int(other is not None and np.array_equal(other, array if array.ndim == 2 else array[..., 0]))
    REPORT["train"] = {"det_train_frames": len(frames), "with_image_and_mask": len(keys),
                       "mask_dir": str(train_mask_dir), "mask_values_in_sample": sorted(values),
                       "mask_modes": sorted(modes), "cv2_equals_pil": f"{cv2_same}/{len(sample_keys)}"}
    log(f"train ground truth: {len(keys)} images (det frames {len(frames)}), mask values {sorted(values)}")
    ok = (REPORT["checks"]["det_train_copies_identical"]
          and REPORT["checks"]["val_labels"]["identical"] == len(ref_labels)
          and REPORT["checks"]["val_masks"]["identical"] == len(ref_masks)
          and all(v["identical"] == v["reference"] for k, v in REPORT["checks"].items() if k.startswith(("val_images", "local_train")))
          and not differ)
    REPORT["all_checks_passed"] = bool(ok)
    (WORK / "verify_report.json").write_text(json.dumps(REPORT, indent=1))
    log(f"ALL CHECKS PASSED: {ok}")


if __name__ == "__main__":
    main()
