"""Write every number quoted in the camera-ready text as a LaTeX macro.

The manuscript never contains a hand-typed result: it reads
outputs/camera_ready/paper/numbers.tex, generated from results.json.
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "camera_ready"
PAPER = OUT / "paper"
SETS = Path(__file__).resolve().parent / "sets"

LINES = []
MISSING = []


NO_SEPARATOR = ("Seed", "Hash", "Commit")  # identifiers, not counts


def macro(name, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        MISSING.append(name)
        value = "\\textbf{??}"
    elif (isinstance(value, str) and re.fullmatch(r"\d+|\d{1,3}(,\d{3})+", value) and len(value.replace(",", "")) >= 4
          and not any(s in name for s in NO_SEPARATOR)):
        # Counts get the same thousands separator as the hand-written numbers ("9{,}500"), also
        # when the caller already formatted them as "8,400" (a bare comma adds space in math mode).
        value = f"{int(value.replace(',', '')):,}".replace(",", "{,}")
    LINES.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def pct(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return f"{100 * value:.{digits}f}"


def count(entry, field="k"):
    if not entry:
        return None
    return str(int(round(entry[field])))


def get(d, *path):
    for key in path:
        if d is None:
            return None
        d = d.get(key) if isinstance(d, dict) else None
    return d


def split_macros(prefix, s):
    rel = s["reliability"]
    macro(f"{prefix}Images", str(s["images"]))
    macro(f"{prefix}Accepted", str(rel["accepted"]))
    for state in ("STABLE", "CHECK", "UNCERTAIN"):
        macro(f"{prefix}{state.capitalize()}", str(rel["states"].get(state, 0)))
    w = rel["stable_ratio_wilson"]
    b = rel["stable_ratio_cluster"]
    macro(f"{prefix}StablePct", pct(w["value"]))
    macro(f"{prefix}StableLow", pct(w["low"]))
    macro(f"{prefix}StableHigh", pct(w["high"]))
    macro(f"{prefix}StableBootLow", pct(b["low"]))
    macro(f"{prefix}StableBootHigh", pct(b["high"]))
    det = s["detection"]
    for key, name in (("precision", "Prec"), ("recall_relevant", "RecRel"), ("recall_all", "RecAll")):
        macro(f"{prefix}{name}", pct(det[key]["value"]))
        macro(f"{prefix}{name}Low", pct(det[key]["low"]))
        macro(f"{prefix}{name}High", pct(det[key]["high"]))
    macro(f"{prefix}RawPrec", pct(get(det, "raw_precision", "value")))
    macro(f"{prefix}RawRecRel", pct(get(det, "raw_recall_relevant", "value")))
    path = s["path"]
    macro(f"{prefix}PathPrec", pct(path["path_precision"]["value"]))
    macro(f"{prefix}PathPrecK", count(path["path_precision"]))
    macro(f"{prefix}PathPrecN", count(path["path_precision"], "n"))
    macro(f"{prefix}PathRec", pct(path["path_recall"]["value"]))
    macro(f"{prefix}PathRecK", count(path["path_recall"]))
    macro(f"{prefix}PathRecN", count(path["path_recall"], "n"))
    conf = path["confusion"]
    macro(f"{prefix}PathOff", str(conf["PATH"]["off"]))
    macro(f"{prefix}SideDirect", str(conf["SIDE"]["direct"]))
    cor = s["corridor"]
    macro(f"{prefix}CorridorIoU", f"{cor['corridor_iou_direct']:.2f}" if cor["corridor_iou_direct"] == cor["corridor_iou_direct"] else None)
    macro(f"{prefix}RoadIoU", f"{cor['road_iou_drivable']:.2f}" if cor["road_iou_drivable"] == cor["road_iou_drivable"] else None)
    adv = s["advisory"]
    nofb = s["advisory_no_fallback"]
    macro(f"{prefix}Occupied", str(adv["occupied_images"]))
    macro(f"{prefix}Unsafe", count(adv["unsafe_clear"]))
    macro(f"{prefix}UnsafeHigh", pct(adv["unsafe_clear"]["high"]))
    macro(f"{prefix}UnsafeNoFB", count(nofb["unsafe_clear"]))
    macro(f"{prefix}Free", count(adv["over_caution"], "n"))
    macro(f"{prefix}Over", count(adv["over_caution"]))
    macro(f"{prefix}OverPct", pct(adv["over_caution"]["value"]))
    macro(f"{prefix}Severe", count(adv["severe"]))
    macro(f"{prefix}SeverePct", pct(adv["severe"]["value"]))
    macro(f"{prefix}Unsupported", count(adv["severe_unsupported"]))
    macro(f"{prefix}UnsupportedPct", pct(get(adv, "severe_unsupported", "value")))
    macro(f"{prefix}Empty", str(adv["no_detection_images"]))
    macro(f"{prefix}Fallback", str(adv["fallback_fired"]))
    macro(f"{prefix}Stop", str(adv["distribution"].get("STOP", 0)))
    macro(f"{prefix}Brake", str(adv["distribution"].get("BRAKE", 0)))
    macro(f"{prefix}Clear", str(adv["distribution"].get("CLEAR", 0)))
    macro(f"{prefix}ClearPct", pct(adv["distribution"].get("CLEAR", 0) / max(s["images"], 1)))
    macro(f"{prefix}MonitorPct", pct(adv["distribution"].get("MONITOR", 0) / max(s["images"], 1)))
    states = rel["by_state"]
    fp = rel["fp_stable_vs_nonstable"]
    macro(f"{prefix}FPStable", pct(fp["stable"]["value"]))
    macro(f"{prefix}FPStableLow", pct(fp["stable"]["low"]))
    macro(f"{prefix}FPStableHigh", pct(fp["stable"]["high"]))
    macro(f"{prefix}FPNonStable", pct(fp["non_stable"]["value"]))
    macro(f"{prefix}FPNonStableLow", pct(fp["non_stable"]["low"]))
    macro(f"{prefix}FPNonStableHigh", pct(fp["non_stable"]["high"]))
    ratio = fp["non_stable"]["value"] / fp["stable"]["value"] if fp["stable"]["value"] else float("nan")
    macro(f"{prefix}FPRatio", f"{ratio:.1f}" if ratio == ratio else None)
    for state in ("STABLE", "CHECK"):
        name = state.capitalize()
        macro(f"{prefix}Trunc{name}", pct(states[state]["truncated_rate"]["value"]))
        macro(f"{prefix}Occl{name}", pct(states[state]["occluded_rate"]["value"]))
        macro(f"{prefix}IoU{name}", f"{states[state]['mean_iou_tp']:.2f}" if states[state]["mean_iou_tp"] == states[state]["mean_iou_tp"] else None)
    for weather, label in (("clear", "Clear"), ("rainy", "Rainy"), ("snowy", "Snowy"), ("foggy", "Foggy")):
        entry = get(s, "by_weather", weather)
        if entry:
            macro(f"{prefix}W{label}Images", str(entry["images"]))
            macro(f"{prefix}W{label}RecRel", pct(entry["recall_relevant"]))
            macro(f"{prefix}W{label}Prec", pct(entry["precision"]))
            macro(f"{prefix}W{label}StablePct", pct(entry["stable_ratio"]["value"]))
            macro(f"{prefix}W{label}Unsafe", count(entry["unsafe_clear"]))
            macro(f"{prefix}W{label}Occupied", count(entry["unsafe_clear"], "n"))
            macro(f"{prefix}W{label}Degraded", pct(entry["degraded_fraction"], 0))
            for group, group_label in (("daytime", "Day"), ("night", "Night")):
                combo = get(s, "by_weather_timeofday", f"{weather}|{group}")
                if combo:
                    macro(f"{prefix}W{label}{group_label}Images", str(combo["images"]))
                    macro(f"{prefix}W{label}{group_label}RecRel", pct(combo["recall_relevant"]))
            night = get(s, "by_weather_timeofday", f"{weather}|night")
            if night and entry["images"]:
                macro(f"{prefix}W{label}NightPct", pct(night["images"] / entry["images"], 0))
    for group, label in (("daytime", "Day"), ("night", "Night")):
        entry = get(s, "by_timeofday", group)
        if entry:
            macro(f"{prefix}{label}Images", str(entry["images"]))
            macro(f"{prefix}{label}StablePct", pct(entry["stable_ratio"]["value"]))
            macro(f"{prefix}{label}Prec", pct(entry["precision"]))
            macro(f"{prefix}{label}RecRel", pct(entry["recall_relevant"]))
            macro(f"{prefix}{label}Unsafe", count(entry["unsafe_clear"]))
            macro(f"{prefix}{label}Occupied", count(entry["unsafe_clear"], "n"))
        cor = get(s, "corridor", "by_timeofday", group)
        if cor:
            macro(f"{prefix}{label}CorridorIoU", f"{cor['corridor_iou_direct']:.2f}")
            macro(f"{prefix}{label}RoadIoU", f"{cor['road_iou_drivable']:.2f}")


VARIANTS = {
    "no_gates": "NoGates",
    "single_region": "Single",
    "no_scene_quality": "NoSQ",
    "no_hybridnets": "NoHyb",
    "no_depth": "NoDepth",
    "B0": "BZero",
    "B1": "BOne",
    "B2": "BTwo",
    "B3": "BThree",
}


def ablation_macros(prefix, results, split):
    entries = get(results, "ablations", split) or {}
    for variant, name in VARIANTS.items():
        entry = entries.get(variant)
        if not entry:
            continue
        adv = entry["advisory_no_fallback"] if variant.startswith("B") else entry["advisory"]
        macro(f"{prefix}{name}Unsafe", count(adv["unsafe_clear"]))
        macro(f"{prefix}{name}Over", count(adv["over_caution"]))
        macro(f"{prefix}{name}OverPct", pct(adv["over_caution"]["value"]))
        macro(f"{prefix}{name}Severe", count(adv["severe"]))
        macro(f"{prefix}{name}Stop", str(adv["distribution"].get("STOP", 0)))
        macro(f"{prefix}{name}Clear", str(adv["distribution"].get("CLEAR", 0)))
        if adv.get("severe_unsupported"):
            macro(f"{prefix}{name}Unsupported", count(adv["severe_unsupported"]))
        macro(f"{prefix}{name}StablePct", pct(entry["reliability"]["stable_ratio_wilson"]["value"]))
        comp = get(results, "comparisons", f"{split}:full_vs_{variant}")
        if comp:
            macro(f"{prefix}{name}Changed", str(comp["advisory_changes"]))
            macro(f"{prefix}{name}OverP", fmt_p(comp["over_caution"]["mcnemar_p"]))
            macro(f"{prefix}{name}UnsafeP", fmt_p(comp["unsafe_clear"]["mcnemar_p"]))
    comp = get(results, "comparisons", f"{split}:full_vs_no_fallback")
    if comp:
        macro(f"{prefix}NoFBChanged", str(comp["advisory_changes"]))
        macro(f"{prefix}NoFBUnsafeP", fmt_p(comp["unsafe_clear"]["mcnemar_p"]))
    for variant, name in (("corridor_road", "Exp"),):
        entry = get(results, "exploratory", f"{split}:{variant}")
        if not entry:
            continue
        adv = entry["advisory"]
        macro(f"{prefix}{name}PathPrec", pct(entry["path"]["path_precision"]["value"]))
        macro(f"{prefix}{name}PathRec", pct(entry["path"]["path_recall"]["value"]))
        macro(f"{prefix}{name}Over", count(adv["over_caution"]))
        macro(f"{prefix}{name}OverPct", pct(adv["over_caution"]["value"]))
        macro(f"{prefix}{name}Severe", count(adv["severe"]))
        macro(f"{prefix}{name}Unsupported", count(adv["severe_unsupported"]))
        macro(f"{prefix}{name}Unsafe", count(adv["unsafe_clear"]))
        comp = get(results, "comparisons", f"{split}:full_vs_{variant}")
        if comp:
            macro(f"{prefix}{name}OverP", fmt_p(comp["over_caution"]["mcnemar_p"]))


def subset_macros(results):
    for key, entry in (results.get("subset_changes") or {}).items():
        split, variant, group = key.split(":")
        prefix = {"hard": "Hard", "standard": "Std", "test": "Test", "train": "Train"}[split]
        name = {"no_depth": "NoDepth", "no_hybridnets": "NoHyb"}[variant]
        label = {"daytime": "Day", "night": "Night"}[group]
        macro(f"{prefix}{name}{label}ChangedPct", pct(entry["fraction"]))
        macro(f"{prefix}{name}{label}Images", str(entry["images"]))


def fmt_p(value):
    """p-value with its relation, for math mode: $p\\TestNoGatesOverP$ gives p<0.001 or p=0.016."""
    if value is None:
        return None
    if value < 0.001:
        return "<0.001"
    return "=" + (f"{value:.3f}" if value < 0.1 else f"{value:.2f}")


def stability_macros(results):
    for block, prefix in (("stability", ""), ("stability_std", "Std"), ("crosshw", "CrossHW")):
        for variant, name in (("full", "Full"), ("no_gates", "NoGates"), ("single_region", "Single"),
                              ("no_scene_quality", "NoSQ"), ("B0", "BZero"), ("B3", "BThree")):
            entry = get(results, block, variant)
            if not entry:
                continue
            macro(f"{prefix}Flip{name}", pct(entry["flip_rate"][0]))
            macro(f"{prefix}Flip{name}Low", pct(entry["flip_rate"][1]))
            macro(f"{prefix}Flip{name}High", pct(entry["flip_rate"][2]))
            macro(f"{prefix}SevereFlip{name}", pct(entry["severe_flip_rate"][0]))
            macro(f"{prefix}StateFlip{name}", pct(entry["state_flip_rate"][0]))
            macro(f"{prefix}Persist{name}", pct(entry["object_persistence"][0]))
            macro(f"{prefix}FlipPairs{name}", str(entry["pairs"]))


def sensitivity_macros(results):
    sens = results.get("sensitivity") or {}
    base = sens.get("default")
    if not base:
        return
    stable0 = base["reliability"]["stable_ratio_wilson"]["value"]
    over0 = base["advisory"]["over_caution"]["value"]
    severe0 = base["advisory"]["severe"]["value"]
    unsafe0 = base["advisory"]["unsafe_clear"]["k"]
    reliability_keys = [k for k in sens if k.startswith("sens:")]
    risk_keys = [k for k in sens if k.startswith("risk:")]
    sq_keys = [k for k in sens if k.startswith("sq:")]

    def max_delta(keys, getter, ref):
        values = [abs(getter(sens[k]) - ref) for k in keys]
        return max(values) if values else None

    macro("DevImages", str(base["images"]))
    macro("DevStablePct", pct(stable0))
    macro("DevOverPct", pct(over0))
    macro("DevSeverePct", pct(severe0))
    macro("DevUnsafe", str(int(unsafe0)))
    macro("DevOccupied", str(base["advisory"]["occupied_images"]))
    macro("SensRelStableDelta", pct(max_delta(reliability_keys, lambda e: e["reliability"]["stable_ratio_wilson"]["value"], stable0)))
    macro("SensRelOverDelta", pct(max_delta(reliability_keys, lambda e: e["advisory"]["over_caution"]["value"], over0)))
    macro("SensRelSevereDelta", pct(max_delta(reliability_keys, lambda e: e["advisory"]["severe"]["value"], severe0)))
    macro("SensRiskOverDelta", pct(max_delta(risk_keys, lambda e: e["advisory"]["over_caution"]["value"], over0)))
    macro("SensRiskSevereDelta", pct(max_delta(risk_keys, lambda e: e["advisory"]["severe"]["value"], severe0)))
    macro("SensSQSevereDelta", pct(max_delta(sq_keys, lambda e: e["advisory"]["severe"]["value"], severe0)))
    unsafe_all = [sens[k]["advisory"]["unsafe_clear"]["k"] for k in reliability_keys + risk_keys + sq_keys]
    macro("SensUnsafeMax", str(int(max(unsafe_all))) if unsafe_all else None)
    macro("SensUnsafeMin", str(int(min(unsafe_all))) if unsafe_all else None)


def operating_macros(results):
    for split, prefix in (("dev", "OpDev"), ("standard", "OpStd")):
        points = sorted(get(results, "operating", split) or [], key=lambda p: p["scale"])
        if not points:
            continue

        def over(p):
            return p["over_caution"] / max(p["free"], 1)

        def unsafe(p):
            return p["unsafe_clear"] / max(p["occupied"], 1)

        base = next((p for p in points if p["scale"] == 1.0), None)
        safe_points = [p for p in points if p["unsafe_clear"] == 0]
        safest = max(safe_points, key=lambda p: p["scale"]) if safe_points else None
        last = points[-1]
        if base:
            macro(f"{prefix}BaseOverPct", pct(over(base)))
            macro(f"{prefix}BaseClearPct", pct(base["clear_images"] / max(base["images"], 1)))
            macro(f"{prefix}BaseSeverePct", pct(base["severe_images"] / max(base["images"], 1)))
        if safest:
            macro(f"{prefix}SafeScale", f"{safest['scale']:g}")
            macro(f"{prefix}SafeOverPct", pct(over(safest)))
            macro(f"{prefix}SafeClearPct", pct(safest["clear_images"] / max(safest["images"], 1)))
        macro(f"{prefix}MaxScale", f"{last['scale']:g}")
        macro(f"{prefix}MaxOverPct", pct(over(last)))
        macro(f"{prefix}MaxUnsafePct", pct(unsafe(last)))
        macro(f"{prefix}Occupied", str(points[0]["occupied"]))
        if base:
            macro(f"{prefix}BaseUnsafe", str(base["unsafe_clear"]))
    # Held-out check: the operating point chosen on dev, read off the 1k test subset.
    dev_points = get(results, "operating", "dev") or []
    std_points = get(results, "operating", "standard") or []
    dev_safe = [p for p in dev_points if p["unsafe_clear"] == 0]
    if dev_safe and std_points:
        scale = max(p["scale"] for p in dev_safe)
        match = next((p for p in std_points if p["scale"] == scale), None)
        if match:
            macro("OpStdAtSafeUnsafe", str(match["unsafe_clear"]))
            macro("OpStdAtSafeOverPct", pct(match["over_caution"] / max(match["free"], 1)))


def operating_check_macros(results):
    """The operating point chosen on dev (largest scale with no unsafe CLEAR there), read off the
    70k train split and the 8,400 held-out val_rest images, with Wilson intervals."""
    from stats import wilson

    dev_safe = [p for p in get(results, "operating", "dev") or [] if p["unsafe_clear"] == 0]
    chosen = max(p["scale"] for p in dev_safe) if dev_safe else None
    if chosen is not None:
        macro("OpChosenScale", f"{chosen:g}")
    for split, prefix in (("train", "OpTrain"), ("valop", "OpValop")):
        points = {p["scale"]: p for p in get(results, "operating", split) or []}
        if 1.0 in points:
            macro(f"{prefix}Images", f"{points[1.0]['images']:,}")
            macro(f"{prefix}Occupied", f"{points[1.0]['occupied']:,}")
            macro(f"{prefix}Free", f"{points[1.0]['free']:,}")
        for scale, tag in ((1.0, "Base"), (chosen, "Chosen")):
            p = points.get(scale)
            if not p:
                continue
            unsafe = wilson(p["unsafe_clear"], p["occupied"])
            over = wilson(p["over_caution"], p["free"])
            macro(f"{prefix}{tag}Unsafe", str(p["unsafe_clear"]))
            for name, value in (("UnsafePct", unsafe[0]), ("UnsafeLow", unsafe[1]), ("UnsafeHigh", unsafe[2])):
                macro(f"{prefix}{tag}{name}", pct(value, 2))
            for name, value in (("OverPct", over[0]), ("OverLow", over[1]), ("OverHigh", over[2])):
                macro(f"{prefix}{tag}{name}", pct(value))
            macro(f"{prefix}{tag}ClearPct", pct(p["clear_images"] / max(p["images"], 1)))


def crosshw_val_macros(results):
    """RTX 3050 vs T4 on the val_rest images processed locally (rerun on the T4 in valop_full)."""
    entry = get(results, "crosshw", "val_rest_full")
    if not entry:
        return
    macro("CrossHWValPairs", f"{entry['pairs']:,}")
    flips = entry["flip_rate"][0] * entry["pairs"]
    assert abs(flips - round(flips)) < 1e-6, f"flip rate x pairs is not a count: {flips}"
    macro("CrossHWValFlips", str(round(flips)))
    macro("CrossHWValFlip", pct(entry["flip_rate"][0], 2))
    macro("CrossHWValFlipHigh", pct(entry["flip_rate"][2], 2))
    macro("CrossHWValSevereFlip", pct(entry["severe_flip_rate"][0], 2))
    macro("CrossHWValStateFlip", pct(entry["state_flip_rate"][0], 2))
    macro("CrossHWValPersist", pct(entry["object_persistence"][0], 1))


def swap_macros(results):
    for stem, name in (("yolo11s", "S"), ("yolo11m", "M")):
        for split, prefix in (("hard", "Hard"), ("standard", "Std"), ("test", "Test")):
            entry = get(results, "swap", f"{split}:{stem}")
            if not entry:
                continue
            macro(f"Swap{name}{prefix}Prec", pct(entry["detection"]["precision"]["value"]))
            macro(f"Swap{name}{prefix}RecRel", pct(entry["detection"]["recall_relevant"]["value"]))
            macro(f"Swap{name}{prefix}Empty", str(entry["advisory"]["no_detection_images"]))
            macro(f"Swap{name}{prefix}Fallback", str(entry["advisory"]["fallback_fired"]))
            macro(f"Swap{name}{prefix}Unsafe", count(entry["advisory"]["unsafe_clear"]))
            macro(f"Swap{name}{prefix}UnsafeNoFB", count(entry["advisory_no_fallback"]["unsafe_clear"]))
            macro(f"Swap{name}{prefix}Occupied", str(entry["advisory"]["occupied_images"]))
            macro(f"Swap{name}{prefix}StablePct", pct(entry["reliability"]["stable_ratio_wilson"]["value"]))


def latency_macros(results):
    rows = [get(results, "latency", split) for split in ("all", "standard", "dev", "hard")]
    rows = [row for row in rows if row]
    if not rows:
        return
    row = rows[0]
    macro("LatImages", str(row["images"]))
    macro("LatHyb", f"{row['hybridnets_s']['median']:.2f}")
    macro("LatDepth", f"{row['depth_s']['median']:.2f}")
    macro("LatYolo", f"{row['yolo_s']['median']:.2f}")
    macro("LatTotal", f"{row['total_s']['median']:.1f}")
    macro("LatTotalPNinety", f"{row['total_s']['p90']:.1f}")
    macro("LatVRAM", f"{row['peak_vram_mb'] / 1024:.1f}")
    t4 = get(results, "latency", "t4")
    if t4:
        macro("LatTFourImages", str(t4["images"]))
        macro("LatTFourHyb", f"{t4['hybridnets_s']['median']:.2f}")
        macro("LatTFourDepth", f"{t4['depth_s']['median']:.2f}")
        macro("LatTFourYolo", f"{t4['yolo_s']['median']:.2f}")
        macro("LatTFourTotal", f"{t4['total_s']['median']:.1f}")
        macro("LatTFourTotalPNinety", f"{t4['total_s']['p90']:.1f}")
        macro("LatTFourVRAM", f"{t4['peak_vram_mb'] / 1024:.1f}")


def kaggle_macros(results):
    info = results.get("kaggle") or {}
    if not info:
        return
    n = info["test_images_on_t4"]
    total = get(results, "splits", "test", "images")
    macro("TestTFourImages", f"{n:,}")
    if isinstance(total, (int, float)) and total:
        macro("TestTFourPct", f"{100 * n / total:.0f}")


def corridor_macros():
    path = OUT / "corridor_stats.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text(encoding="utf-8"))
    for split, prefix in (("hard", "Hard"), ("standard", "Std"), ("test", "Test")):
        entry = stats.get(split)
        if not entry:
            continue
        macro(f"{prefix}LaneGuidedPct", pct(entry["mean_lane_guided_rows"]))
        macro(f"{prefix}MostlyTrapezoidPct", pct(entry["images_mostly_trapezoid"], 0))
        macro(f"{prefix}NoLaneRowsPct", pct(entry["images_no_lane_rows"], 0))


def unsafe_clear_macros():
    """What happened on the unsafe-CLEAR images, read from the evaluation CSVs: the relevant
    direct-lane road user was among the accepted detections (then how the corridor related it),
    or the detector missed it."""
    for split, prefix in (("test", "Test"), ("train", "Train")):
        folder = OUT / "eval" / f"{split}_full"
        if not (folder / "images.csv").exists():
            continue
        with (folder / "images.csv").open(encoding="utf-8", newline="") as handle:
            unsafe = {row["key"]: row for row in csv.DictReader(handle) if row["unsafe_clear"] == "1"}
        hits = {key: [] for key in unsafe}
        with (folder / "objects.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["key"] in hits and row["tp"] == "1" and row["gt_lane"] == "direct" and row["gt_relevant"] == "1":
                    hits[row["key"]].append(row)
        detected = [key for key, found in hits.items() if found]
        missed = [key for key, found in hits.items() if not found]
        macro(f"{prefix}UnsafeDetected", str(len(detected)))
        macro(f"{prefix}UnsafeDetectedSide",
              str(sum(1 for key in detected if all(o["relation"] == "SIDE" for o in hits[key]))))
        macro(f"{prefix}UnsafeDetectedStable",
              str(sum(1 for key in detected if all(o["state"] == "STABLE" for o in hits[key]))))
        macro(f"{prefix}UnsafeMissed", str(len(missed)))
        macro(f"{prefix}UnsafeMissedNight", str(sum(1 for key in missed if unsafe[key]["timeofday"] == "night")))


def provenance_macros():
    manifest = json.loads((SETS / "splits_manifest.json").read_text(encoding="utf-8"))
    for split, name in (("hard", "Hard"), ("dev", "Dev"), ("standard", "Std"), ("test", "Test"), ("val", "Val")):
        if split in manifest["splits"]:
            macro(f"{name}IDsHash", manifest["splits"][split]["ids_sha256"][:16])
    macro("SplitSeed", str(manifest["seed"]))
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    macro("CodeCommit", commit)


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    # The held-out val_rest rerun (valop) is only read for its operating points (OpValop*) and the
    # cross-hardware check (CrossHWVal*); its full split summary stays in results.json (it has no
    # corridor or raw-detector evaluation, so split macros would be incomplete).
    for split, prefix in (("hard", "Hard"), ("standard", "Std"), ("dev", "DevSplit"), ("test", "Test"),
                          ("train", "Train")):
        if split in results["splits"]:
            split_macros(prefix, results["splits"][split])
    ablation_macros("Hard", results, "hard")
    ablation_macros("Std", results, "standard")
    ablation_macros("Dev", results, "dev")
    ablation_macros("Test", results, "test")
    ablation_macros("Train", results, "train")
    operating_check_macros(results)
    crosshw_val_macros(results)
    stability_macros(results)
    subset_macros(results)
    sensitivity_macros(results)
    operating_macros(results)
    swap_macros(results)
    latency_macros(results)
    kaggle_macros(results)
    corridor_macros()
    unsafe_clear_macros()
    provenance_macros()
    PAPER.mkdir(parents=True, exist_ok=True)
    header = "% Generated by camera_ready/paper_numbers.py from results.json. Do not edit.\n"
    (PAPER / "numbers.tex").write_text(header + "\n".join(LINES) + "\n", encoding="utf-8")
    print(f"[NUMBERS] {len(LINES)} macros, missing={len(MISSING)}")
    if MISSING:
        print("  missing:", ", ".join(MISSING[:40]))


if __name__ == "__main__":
    main()
