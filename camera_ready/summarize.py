"""Aggregate ground-truth evaluation CSVs into metrics with confidence intervals."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stats import cluster_bootstrap_ratio, mcnemar_exact, paired_bootstrap_difference, wilson

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "camera_ready" / "eval"


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_eval(name):
    directory = EVAL_DIR / name
    images = read_csv(directory / "images.csv")
    objects = read_csv(directory / "objects.csv") if (directory / "objects.csv").exists() else []
    return images, objects


def as_int(value):
    return int(float(value)) if value not in ("", None) else 0


def per_image(objects, keys, predicate):
    counts = Counter(row["key"] for row in objects if predicate(row))
    return np.asarray([counts.get(key, 0) for key in keys], dtype=np.float64)


def proportion(k, n):
    p, low, high = wilson(k, n)
    return {"k": int(k), "n": int(n), "value": p, "low": low, "high": high}


def clustered(num, den):
    value, low, high = cluster_bootstrap_ratio(num, den)
    return {"k": float(np.sum(num)), "n": float(np.sum(den)), "value": value, "low": low, "high": high}


def detection_metrics(images, objects):
    keys = [row["key"] for row in images]
    n_pred = np.asarray([as_int(row["n_pred"]) for row in images], dtype=np.float64)
    n_tp = np.asarray([as_int(row["n_tp"]) for row in images], dtype=np.float64)
    n_gt = np.asarray([as_int(row["n_gt"]) for row in images], dtype=np.float64)
    n_gt_rel = np.asarray([as_int(row["n_gt_relevant"]) for row in images], dtype=np.float64)
    n_tp_rel = np.asarray([as_int(row["n_tp_relevant"]) for row in images], dtype=np.float64)
    out = {
        "precision": clustered(n_tp, n_pred),
        "recall_all": clustered(n_tp, n_gt),
        "recall_relevant": clustered(n_tp_rel, n_gt_rel),
    }
    p, r = out["precision"]["value"], out["recall_relevant"]["value"]
    out["f1_relevant"] = 2 * p * r / (p + r) if p + r > 0 else float("nan")
    if images and images[0].get("raw_n", "") != "":
        raw_n = np.asarray([as_int(row["raw_n"]) for row in images], dtype=np.float64)
        raw_tp = np.asarray([as_int(row["raw_tp"]) for row in images], dtype=np.float64)
        raw_tp_rel = np.asarray([as_int(row["raw_tp_relevant"]) for row in images], dtype=np.float64)
        out["raw_precision"] = clustered(raw_tp, raw_n)
        out["raw_recall_relevant"] = clustered(raw_tp_rel, n_gt_rel)
    return out


def reliability_metrics(images, objects):
    keys = [row["key"] for row in images]
    total = per_image(objects, keys, lambda r: True)
    stable = per_image(objects, keys, lambda r: r["state"] == "STABLE")
    out = {
        "accepted": int(total.sum()),
        "states": dict(Counter(row["state"] for row in objects)),
        "stable_ratio_wilson": proportion(int(stable.sum()), int(total.sum())),
        "stable_ratio_cluster": clustered(stable, total),
        "by_state": {},
    }
    for state in ("STABLE", "CHECK", "UNCERTAIN"):
        rows = [row for row in objects if row["state"] == state]
        state_total = per_image(objects, keys, lambda r, s=state: r["state"] == s)
        state_fp = per_image(objects, keys, lambda r, s=state: r["state"] == s and r["tp"] == "0")
        tps = [row for row in rows if row["tp"] == "1"]
        tp_total = per_image(objects, keys, lambda r, s=state: r["state"] == s and r["tp"] == "1")
        occluded = per_image(
            objects, keys, lambda r, s=state: r["state"] == s and r["tp"] == "1" and r["gt_occluded"] == "1"
        )
        truncated = per_image(
            objects, keys, lambda r, s=state: r["state"] == s and r["tp"] == "1" and r["gt_truncated"] == "1"
        )
        out["by_state"][state] = {
            "n": len(rows),
            "fp_rate": clustered(state_fp, state_total),
            "occluded_rate": clustered(occluded, tp_total),
            "truncated_rate": clustered(truncated, tp_total),
            "mean_iou_tp": float(np.mean([float(row["iou"]) for row in tps])) if tps else float("nan"),
        }
    non_stable_total = total - stable
    non_stable_fp = per_image(objects, keys, lambda r: r["state"] != "STABLE" and r["tp"] == "0")
    stable_fp = per_image(objects, keys, lambda r: r["state"] == "STABLE" and r["tp"] == "0")
    out["fp_stable_vs_nonstable"] = {
        "stable": clustered(stable_fp, stable),
        "non_stable": clustered(non_stable_fp, non_stable_total),
    }
    return out


def path_metrics(objects):
    tp = [row for row in objects if row["tp"] == "1"]
    labels_pred = ("PATH", "EDGE", "SIDE")
    labels_gt = ("direct", "alternative", "off")
    confusion = {p: {g: 0 for g in labels_gt} for p in labels_pred}
    for row in tp:
        if row["relation"] in confusion and row["gt_lane"] in labels_gt:
            confusion[row["relation"]][row["gt_lane"]] += 1
    pred_path = sum(confusion["PATH"].values())
    gt_direct = sum(confusion[p]["direct"] for p in labels_pred)
    hit = confusion["PATH"]["direct"]
    precision = proportion(hit, pred_path)
    recall = proportion(hit, gt_direct)
    f1 = (
        2 * precision["value"] * recall["value"] / (precision["value"] + recall["value"])
        if pred_path and gt_direct and (precision["value"] + recall["value"]) > 0
        else float("nan")
    )
    side_ok = confusion["SIDE"]["off"] + confusion["SIDE"]["alternative"]
    return {
        "confusion": confusion,
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": f1,
        "side_not_direct": proportion(side_ok, sum(confusion["SIDE"].values())),
    }


def advisory_metrics(images, column="advisory"):
    occupied = [row for row in images if row["occupied"] == "1"]
    free_any = [row for row in images if row["occupied_any"] == "0"]
    unsafe_col = "unsafe_clear" if column == "advisory" else "unsafe_clear_nofb"
    unsafe = sum(as_int(row[unsafe_col]) for row in occupied)
    severity = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}
    over = sum(1 for row in free_any if severity[row[column]] >= 2)
    severe_rows = [row for row in images if row[column] in {"BRAKE", "STOP"}]
    unsupported = sum(as_int(row["severe_unsupported"]) for row in severe_rows) if column == "advisory" else None
    return {
        "distribution": dict(Counter(row[column] for row in images)),
        "occupied_images": len(occupied),
        "unsafe_clear": proportion(unsafe, len(occupied)),
        "over_caution": proportion(over, len(free_any)),
        "severe": proportion(len(severe_rows), len(images)),
        "severe_unsupported": proportion(unsupported, len(severe_rows)) if unsupported is not None else None,
        "fallback_fired": sum(as_int(row["fallback_fired"]) for row in images) if column == "advisory" else 0,
        "no_detection_images": sum(1 for row in images if as_int(row["n_pred"]) == 0),
    }


def corridor_metrics(images):
    def mean_of(rows, field):
        values = [float(row[field]) for row in rows if row.get(field, "") not in ("", "nan")]
        values = [v for v in values if v == v]
        return float(np.mean(values)) if values else float("nan")

    groups = defaultdict(list)
    for row in images:
        groups[row["timeofday"]].append(row)
    return {
        "corridor_iou_direct": mean_of(images, "corridor_iou"),
        "road_iou_drivable": mean_of(images, "road_iou"),
        "by_timeofday": {
            key: {"n": len(rows), "corridor_iou_direct": mean_of(rows, "corridor_iou"), "road_iou_drivable": mean_of(rows, "road_iou")}
            for key, rows in sorted(groups.items())
        },
    }


def stratified(images, objects, field):
    """Metrics per value of an image field; a tuple of fields groups by their joined values ("clear|night")."""
    fields = field if isinstance(field, tuple) else (field,)
    groups = defaultdict(list)
    for row in images:
        groups["|".join(row[name] or "none" for name in fields)].append(row)
    result = {}
    for value, rows in sorted(groups.items()):
        keys = {row["key"] for row in rows}
        objs = [row for row in objects if row["key"] in keys]
        det = detection_metrics(rows, objs)
        stable = sum(1 for row in objs if row["state"] == "STABLE")
        adv = advisory_metrics(rows)
        result[value] = {
            "images": len(rows),
            "objects": len(objs),
            "stable_ratio": proportion(stable, len(objs)),
            "precision": det["precision"]["value"],
            "recall_relevant": det["recall_relevant"]["value"],
            "mean_scene_score": float(np.mean([float(row["scene_score"]) for row in rows])),
            "degraded_fraction": float(np.mean([row["scene_status"] == "DEGRADED" for row in rows])),
            "unsafe_clear": adv["unsafe_clear"],
            "unsafe_clear_nofb": advisory_metrics(rows, "advisory_nofb")["unsafe_clear"],
            "over_caution": adv["over_caution"],
            "advisory_distribution": adv["distribution"],
        }
    return result


def summarize(name):
    images, objects = load_eval(name)
    return {
        "name": name,
        "images": len(images),
        "detection": detection_metrics(images, objects),
        "reliability": reliability_metrics(images, objects),
        "path": path_metrics(objects),
        "advisory": advisory_metrics(images),
        "advisory_no_fallback": advisory_metrics(images, "advisory_nofb"),
        "corridor": corridor_metrics(images),
        "by_timeofday": stratified(images, objects, "timeofday"),
        "by_category": stratified(images, objects, "category") if images and images[0]["category"] else {},
        "by_weather": stratified(images, objects, "weather"),
        # Weather tags are confounded with time of day (most night images are tagged clear).
        "by_weather_timeofday": stratified(images, objects, ("weather", "timeofday")),
    }


def compare(base_name, other_name, base_column="advisory", other_column="advisory"):
    """Paired comparison of image-level outcomes between two evaluations."""
    base, _ = load_eval(base_name)
    other, _ = load_eval(other_name)
    other_by_key = {row["key"]: row for row in other}
    pairs = [(row, other_by_key[row["key"]]) for row in base if row["key"] in other_by_key]
    severity = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}

    def unsafe(row, column):
        return row["occupied"] == "1" and row[column] == "CLEAR"

    def over(row, column):
        return row["occupied_any"] == "0" and severity[row[column]] >= 2

    result = {}
    for metric, func in (("unsafe_clear", unsafe), ("over_caution", over)):
        a = [func(x, base_column) for x, _ in pairs]
        b = [func(y, other_column) for _, y in pairs]
        b_count, c_count, p_value = mcnemar_exact(a, b)
        diff = paired_bootstrap_difference(np.asarray(a, float), np.asarray(b, float))
        result[metric] = {
            "base_count": int(sum(a)),
            "other_count": int(sum(b)),
            "base_only": b_count,
            "other_only": c_count,
            "mcnemar_p": p_value,
            "mean_diff": diff,
        }
    flips = sum(1 for x, y in pairs if x[base_column] != y[other_column])
    result["advisory_changes"] = flips
    result["pairs"] = len(pairs)
    return result


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
