"""Manual audit kits for the authors (optional, about 15-30 minutes per rater).

1. lane_audit:     60 ground-truth lane assignments (20 direct, 20 alternative,
                   20 off), sampled with a fixed seed from the dev split. Each panel
                   shows the object box (white), the ground-contact strip (yellow)
                   and the annotated drivable area (green = direct, blue =
                   alternative). Raters mark in lane_audit.csv whether the automatic
                   label is right.
2. advisory_audit: 100 dev images with the system's boxes and global advisory.
                   Raters mark each advisory as acceptable / too cautious / not
                   cautious enough in advisory_audit.csv.

The dev split is used so that the audit never touches test images.
``python camera_ready/audit_kit.py analyse`` summarises the filled CSVs
(agreement rate with Wilson CI; Cohen's kappa when two raters' files exist).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gt_eval import ground_lane, load_drivable, load_gt, read_ids  # noqa: E402
from stats import wilson  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "outputs" / "camera_ready"
AUDIT = OUT / "audit"
VAL = ROOT.parent / "data" / "images" / "100k" / "val"
SEED = 20261109


def strip_box(box, shape):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx1, cx2 = int(np.clip(x1 + 0.25 * bw, 0, width - 1)), int(np.clip(x2 - 0.25 * bw, 0, width))
    ry1 = int(np.clip(y2 - 2, 0, height - 1))
    ry2 = int(np.clip(y2 + max(4.0, 0.08 * bh), ry1 + 1, height))
    return cx1, ry1, cx2, ry2


def lane_panel(key, obj, drivable):
    image = cv2.imread(str(VAL / f"{key}.jpg"))
    overlay = image.copy()
    overlay[drivable == 0] = (0.55 * overlay[drivable == 0] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
    overlay[drivable == 1] = (0.55 * overlay[drivable == 1] + 0.45 * np.array([220, 120, 40])).astype(np.uint8)
    x1, y1, x2, y2 = map(int, obj["box"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    sx1, sy1, sx2, sy2 = strip_box(obj["box"], image.shape)
    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (0, 230, 255), 2)
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * 1.2 + 40)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
    crop = overlay[cy1:cy2, cx1:cx2]
    scale = 320 / max(crop.shape[:2])
    return cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))


def sheet(panels, labels, path, columns=4):
    tiles = []
    for panel, label in zip(panels, labels):
        canvas = np.full((360, 340, 3), 255, np.uint8)
        canvas[30:30 + panel.shape[0], 10:10 + panel.shape[1]] = panel
        cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        tiles.append(canvas)
    while len(tiles) % columns:
        tiles.append(np.full((360, 340, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.vstack(rows))


def make_lane_audit():
    ids = read_ids(HERE / "sets" / "dev_ids.txt")
    gt = load_gt(ids)
    pool = {"direct": [], "alternative": [], "off": []}
    for key in ids:
        drivable = load_drivable(gt[key])
        for index, obj in enumerate(gt[key]["objects"]):
            if obj["box"][3] - obj["box"][1] < 0.05 * gt[key]["height"]:
                continue  # audit relevant objects only
            pool[ground_lane(drivable, obj["box"])].append((key, index))
    rng = np.random.default_rng(SEED)
    picks = []
    for lane in ("direct", "alternative", "off"):
        items = pool[lane]
        chosen = rng.choice(len(items), size=min(20, len(items)), replace=False)
        picks += [(lane, *items[i]) for i in chosen]
    order = rng.permutation(len(picks))
    picks = [picks[i] for i in order]
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows, panels, labels = [], [], []
    for number, (lane, key, index) in enumerate(picks, start=1):
        obj = gt[key]["objects"][index]
        panels.append(lane_panel(key, obj, load_drivable(gt[key])))
        labels.append(f"#{number} auto: {lane}")
        rows.append({"id": number, "image": key, "object_index": index, "auto_label": lane,
                     "human_label (direct/alternative/off)": ""})
    for start in range(0, len(panels), 12):
        sheet(panels[start:start + 12], labels[start:start + 12], AUDIT / f"lane_audit_{start // 12 + 1:02d}.jpg")
    with (AUDIT / "lane_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AUDIT] lane audit: {len(rows)} objects, {math.ceil(len(panels) / 12)} sheets")


def make_advisory_audit():
    ids = read_ids(HERE / "sets" / "dev_ids.txt")
    rng = np.random.default_rng(SEED + 1)
    chosen = sorted(rng.choice(len(ids), size=100, replace=False))
    colors = {"PATH": (40, 100, 235), "EDGE": (120, 175, 30), "SIDE": (215, 120, 40)}
    out = AUDIT / "advisory_images"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, i in enumerate(chosen, start=1):
        key = ids[i]
        report = json.loads((OUT / "runs" / "dev_full" / f"upgraded_{key}.json").read_text(encoding="utf-8"))
        image = cv2.imread(str(VAL / f"{key}.jpg"))
        for det in report["detections"]:
            x1, y1, x2, y2 = det["box_xyxy"]
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[det["corridor_relation"]], 2)
        banner = f"#{number}  advisory: {report['global_advisory']}  (critical: {report['critical_object']})"
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(image, banner[:95], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.imwrite(str(out / f"{number:03d}_{key}.jpg"), image)
        rows.append({"id": number, "image": key, "system_advisory": report["global_advisory"],
                     "rating (acceptable/too_cautious/not_cautious_enough)": ""})
    with (AUDIT / "advisory_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AUDIT] advisory audit: {len(rows)} images")


def make_unsafe_sheet(eval_name="test_full", run_name="test_full"):
    """Contact sheet of every unsafe-CLEAR image, for the authors to inspect."""
    rows = [r for r in csv.DictReader((OUT / "eval" / eval_name / "images.csv").open(encoding="utf-8"))
            if r["unsafe_clear"] == "1"]
    if not rows:
        print("[AUDIT] no unsafe-CLEAR images")
        return
    keys = [r["key"] for r in rows]
    gt = load_gt(keys)
    tiles = []
    for number, row in enumerate(rows, start=1):
        key = row["key"]
        image = cv2.imread(str(VAL / f"{key}.jpg"))
        drivable = load_drivable(gt[key])
        image[drivable == 0] = (0.6 * image[drivable == 0] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
        for obj in gt[key]["objects"]:
            x1, y1, x2, y2 = map(int, obj["box"])
            lane = ground_lane(drivable, obj["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(image, lane[:3], (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        report = json.loads((OUT / "runs" / run_name / f"upgraded_{key}.json").read_text(encoding="utf-8"))
        for det in report["detections"]:
            x1, y1, x2, y2 = det["box_xyxy"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.rectangle(image, (0, 0), (image.shape[1], 36), (255, 255, 255), -1)
        cv2.putText(image, f"#{number} {key} {row['timeofday']} {row['weather']} | system: {report['global_advisory']}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        tiles.append(cv2.resize(image, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.full((360, 640, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    AUDIT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(AUDIT / "unsafe_clear_test.jpg"), grid)
    with (AUDIT / "unsafe_clear_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image", "timeofday", "weather", "author_note (real miss / annotation edge case / other)"])
        for number, row in enumerate(rows, start=1):
            writer.writerow([number, row["key"], row["timeofday"], row["weather"], ""])
    print(f"[AUDIT] unsafe-CLEAR sheet: {len(rows)} images")


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def analyse():
    result = {}
    lane_files = sorted(AUDIT.glob("lane_audit*.csv"))
    for path in lane_files:
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("human_label (direct/alternative/off)", "").strip()]
        if rows:
            agree = sum(r["auto_label"] == r["human_label (direct/alternative/off)"].strip() for r in rows)
            result[path.name] = {"rated": len(rows), "agreement": wilson(agree, len(rows))}
    advisory_files = sorted(AUDIT.glob("advisory_audit*.csv"))
    ratings = {}
    for path in advisory_files:
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r.get("rating (acceptable/too_cautious/not_cautious_enough)", "").strip()]
        if rows:
            values = [r["rating (acceptable/too_cautious/not_cautious_enough)"].strip() for r in rows]
            ratings[path.name] = values
            result[path.name] = {v: values.count(v) for v in sorted(set(values))}
    names = list(ratings)
    if len(names) >= 2 and len(ratings[names[0]]) == len(ratings[names[1]]):
        result["advisory_kappa"] = cohen_kappa(ratings[names[0]], ratings[names[1]])
    (AUDIT / "audit_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("make", "analyse", "unsafe"))
    args = parser.parse_args()
    if args.command == "make":
        make_lane_audit()
        make_advisory_audit()
    elif args.command == "unsafe":
        make_unsafe_sheet()
    else:
        analyse()


if __name__ == "__main__":
    main()
