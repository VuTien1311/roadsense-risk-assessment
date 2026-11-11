"""Ground-truth evaluation of pipeline reports against BDD100K val labels.

Labels: BDD100K val detection boxes (det_20 categories, with ``occluded`` /
``truncated`` flags), per-image ``timeofday`` / ``weather`` / ``scene`` tags and
drivable-area maps (0 = direct / ego lane, 1 = alternative lane, 2 = background),
read from the FiftyOne export ``samples.json`` + ``fields/drivable/*.png``.

Definitions (fixed before looking at any result)
-----------------------------------------------
road user      GT box whose category is a traffic participant (pedestrian, rider,
               other person, car, truck, bus, train, motorcycle, bicycle, other
               vehicle, trailer). Traffic lights/signs are not road users.
relevant       road user whose box height >= 5 % of the image height (36 px at
               720p; roughly within 40 m for a 1.5 m tall car at the default
               focal length used by the pipeline).
GT lane        drivable label under the object's ground-contact strip: rows
               [y2-2, y2+max(4, 0.08 h)], central half of the box width. The
               object is ``direct`` / ``alternative`` when those labels cover at
               least 25 % of the strip (majority wins, ties -> direct), else ``off``.
match          greedy by confidence, class-agnostic over road users, IoU >= 0.5.
occupied       image has a relevant road user whose GT lane is ``direct``.
unsafe CLEAR   global advisory is CLEAR while the image is occupied.
over-caution   global advisory is SLOW/BRAKE/STOP while no road user of any
               size is in the direct lane.
unsupported    global advisory is BRAKE/STOP but no object that carries that
severe         advisory is a true positive inside the direct or alternative lane.
no-fallback    the advisory obtained by disabling the depth-only fallback, i.e.
               CLEAR whenever no object was accepted (exact, because the fallback
               only acts when the accepted object list is empty).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

LABEL_DIR = ROOT.parent / "data" / "bdd100k_labels_val"
EVAL_DIR = ROOT / "outputs" / "camera_ready" / "eval"
SETS_DIR = HERE / "sets"
ROAD_USERS = {
    "pedestrian",
    "rider",
    "other person",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "other vehicle",
    "trailer",
}
RELEVANT_HEIGHT_FRACTION = 0.05
IOU_THRESHOLD = 0.5
DRIVABLE_NAMES = {0: "direct", 1: "alternative", 2: "off"}
SEVERITY = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}
_GT_CACHE = {}


def classification(sample, field):
    value = sample.get(field) or {}
    return value.get("label") or "undefined"


def load_gt(ids=None):
    """Return {image_id: gt_record}. Cached in-process."""
    if "all" not in _GT_CACHE:
        payload = json.loads((LABEL_DIR / "samples.json").read_text(encoding="utf-8"))
        records = {}
        for sample in payload["samples"]:
            key = Path(sample["filepath"]).stem
            width = sample["metadata"]["width"]
            height = sample["metadata"]["height"]
            objects = []
            for det in (sample.get("detections") or {}).get("detections", []) or []:
                if det["label"] not in ROAD_USERS:
                    continue
                x, y, w, h = det["bounding_box"]
                objects.append(
                    {
                        "label": det["label"],
                        "box": [x * width, y * height, (x + w) * width, (y + h) * height],
                        "occluded": bool(det.get("occluded", False)),
                        "truncated": bool(det.get("truncated", False)),
                    }
                )
            records[key] = {
                "width": width,
                "height": height,
                "objects": objects,
                "timeofday": classification(sample, "timeofday"),
                "weather": classification(sample, "weather"),
                "scene": classification(sample, "scene"),
                "drivable_path": LABEL_DIR / sample["drivable"]["mask_path"],
            }
        _GT_CACHE["all"] = records
    records = _GT_CACHE["all"]
    if ids is None:
        return records
    return {key: records[key] for key in ids if key in records}


def load_drivable(record):
    mask = cv2.imread(str(record["drivable_path"]), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(record["drivable_path"])
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def ground_lane(drivable, box):
    height, width = drivable.shape
    x1, y1, x2, y2 = box
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    cx1 = int(np.clip(x1 + 0.25 * box_w, 0, width - 1))
    cx2 = int(np.clip(x2 - 0.25 * box_w, 0, width))
    if cx2 - cx1 < 3:
        center = int(np.clip((x1 + x2) / 2.0, 1, width - 2))
        cx1, cx2 = center - 1, center + 2
    ry1 = int(np.clip(y2 - 2, 0, height - 1))
    ry2 = int(np.clip(y2 + max(4.0, 0.08 * box_h), ry1 + 1, height))
    strip = drivable[ry1:ry2, cx1:cx2]
    if strip.size == 0:
        return "off"
    direct = float(np.mean(strip == 0))
    alternative = float(np.mean(strip == 1))
    if direct + alternative < 0.25:
        return "off"
    return "direct" if direct >= alternative else "alternative"


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes, pred_scores, gt_boxes, threshold=IOU_THRESHOLD):
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    used = set()
    matches = [None] * len(pred_boxes)
    for i in order:
        best_j, best_iou = None, threshold
        for j, gt_box in enumerate(gt_boxes):
            if j in used:
                continue
            value = box_iou(pred_boxes[i], gt_box)
            if value >= best_iou and (best_j is None or value > best_iou):
                best_j, best_iou = j, value
        if best_j is not None:
            used.add(best_j)
            matches[i] = (best_j, best_iou)
    return matches


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_reports(run_dir):
    reports = {}
    for path in Path(run_dir).glob("upgraded_*.json"):
        key = path.stem.replace("upgraded_", "", 1)
        reports[key] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def no_fallback_advisory(report):
    return "CLEAR" if report["object_count"] == 0 else report["global_advisory"]


def corridor_metrics(cache_dir, key, drivable):
    path = Path(cache_dir) / "hybridnets" / f"{key}.npz"
    if not path.exists():
        return None, None
    import test3  # imported lazily; needs no model weights

    data = np.load(path)
    road, lane = data["road"], data["lane"]
    corridor = test3.ADASPipeline.get_ego_corridor(road, lane) > 0
    direct = drivable == 0
    drivable_any = drivable <= 1
    road_bool = road > 0

    def iou(a, b):
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else float("nan")

    return iou(corridor, direct), iou(road_bool, drivable_any)


_BOXES = None  # {key: [{"box", "confidence", "class_name"}]} from harness --yolo-boxes files


def load_boxes(paths):
    boxes = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    boxes[item["key"]] = item["detections"]
    return boxes


def raw_detections(cache_dir, yolo_tag, key):
    if _BOXES is not None:
        if key not in _BOXES:
            return None
        return [{"box": d["box"], "confidence": d["confidence"], "class": d["class_name"]} for d in _BOXES[key]]
    path = Path(cache_dir) / "yolo" / yolo_tag / f"{key}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return [
        {"box": [float(v) for v in item["box"]], "confidence": float(item["confidence"]), "class": item["class_name"]}
        for item in payload["detections"]
    ]


def evaluate_run(run_dir, split, cache_dir=None, yolo_tag=None, with_corridor=False):
    reports = load_reports(run_dir)
    split_ids = read_ids(SETS_DIR / f"{split}_ids.txt") if split else sorted(reports)
    categories = {}
    if split == "hard":
        categories = json.loads((SETS_DIR / "hard_categories.json").read_text(encoding="utf-8"))
    gt = load_gt(split_ids)
    image_rows, object_rows = [], []
    missing = [key for key in split_ids if key not in reports]
    for key in split_ids:
        if key not in reports or key not in gt:
            continue
        report = reports[key]
        record = gt[key]
        drivable = load_drivable(record)
        height = record["height"]
        gt_objects = record["objects"]
        for obj in gt_objects:
            obj["lane"] = ground_lane(drivable, obj["box"])
            obj["height_px"] = obj["box"][3] - obj["box"][1]
            obj["relevant"] = obj["height_px"] >= RELEVANT_HEIGHT_FRACTION * height
        gt_boxes = [obj["box"] for obj in gt_objects]
        preds = report["detections"]
        pred_boxes = [[float(v) for v in det["box_xyxy"]] for det in preds]
        matches = greedy_match(pred_boxes, [det["confidence"] for det in preds], gt_boxes)

        matched_gt = set()
        for det, match in zip(preds, matches):
            row = {
                "key": key,
                "class": det["class"],
                "confidence": det["confidence"],
                "state": det["original_unit_status"],
                "c_u": det["original_unit_confidence"],
                "c_d": det["distance_confidence"],
                "u": det["original_inverse_depth_unit"],
                "relation": det["corridor_relation"],
                "advisory": det["advisory"],
                "risk": det["risk_score"],
                "tp": int(match is not None),
                "iou": match[1] if match else 0.0,
                "gt_label": "",
                "gt_lane": "",
                "gt_occluded": "",
                "gt_truncated": "",
                "gt_relevant": "",
            }
            if match:
                obj = gt_objects[match[0]]
                matched_gt.add(match[0])
                row.update(
                    {
                        "gt_label": obj["label"],
                        "gt_lane": obj["lane"],
                        "gt_occluded": int(obj["occluded"]),
                        "gt_truncated": int(obj["truncated"]),
                        "gt_relevant": int(obj["relevant"]),
                    }
                )
            object_rows.append(row)

        raw = raw_detections(cache_dir, yolo_tag, key) if (cache_dir and yolo_tag) or _BOXES is not None else None
        raw_tp = raw_n = None
        if raw is not None:
            raw_matches = greedy_match([r["box"] for r in raw], [r["confidence"] for r in raw], gt_boxes)
            raw_tp = sum(m is not None for m in raw_matches)
            raw_n = len(raw)
            raw_matched_relevant = sum(
                1 for m in raw_matches if m is not None and gt_objects[m[0]]["relevant"]
            )
        occupied = any(o["relevant"] and o["lane"] == "direct" for o in gt_objects)
        occupied_any = any(o["lane"] == "direct" for o in gt_objects)
        advisory = report["global_advisory"]
        advisory_nofb = no_fallback_advisory(report)
        severe = advisory in {"BRAKE", "STOP"}
        supported = False
        if severe:
            for det, match in zip(preds, matches):
                if det["advisory"] == advisory and match is not None:
                    if gt_objects[match[0]]["lane"] in {"direct", "alternative"}:
                        supported = True
        corridor_iou = road_iou = None
        if with_corridor and cache_dir:
            corridor_iou, road_iou = corridor_metrics(cache_dir, key, drivable)
        image_rows.append(
            {
                "key": key,
                "category": categories.get(key, ""),
                "timeofday": record["timeofday"],
                "weather": record["weather"],
                "scene": record["scene"],
                "scene_status": report["scene_quality"]["status"],
                "scene_score": report["scene_quality"]["score"],
                "depth_scan": report["depth_corridor_scan"]["status"],
                "n_gt": len(gt_objects),
                "n_gt_relevant": sum(o["relevant"] for o in gt_objects),
                "n_pred": len(preds),
                "n_tp": sum(m is not None for m in matches),
                "n_tp_relevant": sum(1 for j in matched_gt if gt_objects[j]["relevant"]),
                "raw_n": raw_n if raw_n is not None else "",
                "raw_tp": raw_tp if raw_tp is not None else "",
                "raw_tp_relevant": raw_matched_relevant if raw is not None else "",
                "n_stable": sum(d["original_unit_status"] == "STABLE" for d in preds),
                "occupied": int(occupied),
                "occupied_any": int(occupied_any),
                "advisory": advisory,
                "advisory_confidence": report["global_advisory_confidence"],
                "advisory_nofb": advisory_nofb,
                "unsafe_clear": int(occupied and advisory == "CLEAR"),
                "unsafe_clear_nofb": int(occupied and advisory_nofb == "CLEAR"),
                "over_caution": int((not occupied_any) and SEVERITY[advisory] >= SEVERITY["SLOW"]),
                "severe": int(severe),
                "severe_unsupported": int(severe and not supported),
                "fallback_fired": int(report["object_count"] == 0 and advisory != "CLEAR"),
                "corridor_iou": corridor_iou if corridor_iou is not None else "",
                "road_iou": road_iou if road_iou is not None else "",
            }
        )
    return image_rows, object_rows, missing


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global LABEL_DIR, SETS_DIR, EVAL_DIR, _BOXES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Directory with upgraded_*.json reports")
    parser.add_argument("--split", required=True, choices=("hard", "dev", "standard", "test", "val", "val_rest", "train"))
    parser.add_argument("--labels", default=str(LABEL_DIR), help="folder with samples.json and fields/drivable/")
    parser.add_argument("--sets-dir", default=str(SETS_DIR), help="folder with <split>_ids.txt")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="output folder (one sub-folder per run)")
    parser.add_argument("--yolo-boxes", default="", help="comma-separated harness --yolo-boxes JSONL files (instead of the YOLO cache)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cache", default=str(ROOT / "outputs" / "camera_ready" / "cache"))
    parser.add_argument("--yolo-tag", default="")
    parser.add_argument("--corridor", action="store_true")
    args = parser.parse_args()
    LABEL_DIR, SETS_DIR, EVAL_DIR = Path(args.labels), Path(args.sets_dir), Path(args.eval_dir)
    if args.yolo_boxes:
        _BOXES = load_boxes([p for p in args.yolo_boxes.split(",") if p])
    images, objects, missing = evaluate_run(
        args.run, args.split, args.cache, args.yolo_tag or None, args.corridor
    )
    out = EVAL_DIR / args.name
    write_csv(out / "images.csv", images)
    write_csv(out / "objects.csv", objects)
    print(f"[EVAL] {args.name}: images={len(images)} objects={len(objects)} missing_reports={len(missing)}")


if __name__ == "__main__":
    main()
