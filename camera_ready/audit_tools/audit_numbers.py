"""Independent recomputation of the paper's headline numbers from the evaluation CSVs.

Uses no project code: only csv + the macro file. Run from outputs/camera_ready:
    python audit_numbers.py <camera_ready output folder>
PASS = the macro printed in the paper equals the value recomputed here (to the macro's own rounding).
"""
import csv
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (OUT / "paper" / "numbers.tex").read_text(encoding="utf-8")))
results = []


def macro(name):
    return MACROS[name].replace("{,}", "").replace("\\,", "")


def rows(run, kind):
    with open(OUT / "eval" / run / f"{kind}.csv", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check(name, value, digits=None):
    """Compare a recomputed value with the macro, at the macro's number of decimals."""
    text = macro(name)
    try:
        shown = float(text)
    except ValueError:
        results.append(False)
        print(f"FAIL {name}: macro {text!r} is not a number (recomputed {value})")
        return
    if digits is None:
        digits = len(text.split(".")[1]) if "." in text else 0
    ok = abs(round(float(value), digits) - shown) < 10 ** (-digits) / 2 + 1e-12
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: paper {text} | recomputed {round(float(value), digits + 2)}")


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


SEVERE = {"BRAKE", "STOP"}

for split, prefix in (("test", "Test"), ("hard", "Hard")):
    images = rows(f"{split}_full", "images")
    objects = rows(f"{split}_full", "objects")
    n = len(images)
    stable = [o for o in objects if o["state"] == "STABLE"]
    other = [o for o in objects if o["state"] != "STABLE"]
    check(f"{prefix}Accepted", len(objects))
    check(f"{prefix}Stable", len(stable))
    check(f"{prefix}StablePct", pct(len(stable), len(objects)))
    occupied = sum(int(r["occupied"]) for r in images)
    unsafe = sum(int(r["unsafe_clear"]) for r in images)
    check(f"{prefix}Occupied", occupied)
    check(f"{prefix}Unsafe", unsafe)
    if split != "test":
        continue
    fp_stable = pct(sum(o["tp"] == "0" for o in stable), len(stable))
    fp_other = pct(sum(o["tp"] == "0" for o in other), len(other))
    check("TestFPStable", fp_stable)
    check("TestFPNonStable", fp_other)
    check("TestFPRatio", fp_other / fp_stable)
    check("TestPrec", pct(sum(o["tp"] == "1" for o in objects), len(objects)))
    check("TestRecRel", pct(sum(int(r["n_tp_relevant"]) for r in images), sum(int(r["n_gt_relevant"]) for r in images)))
    check("TestRawPrec", pct(sum(int(r["raw_tp"]) for r in images), sum(int(r["raw_n"]) for r in images)))
    check("TestRawRecRel", pct(sum(int(r["raw_tp_relevant"]) for r in images), sum(int(r["n_gt_relevant"]) for r in images)))
    clear = sum(r["advisory"] == "CLEAR" for r in images)
    severe = sum(r["advisory"] in SEVERE for r in images)
    check("TestClear", clear)
    check("TestClearPct", pct(clear, n))
    check("TestSeverePct", pct(severe, n))
    check("TestStop", sum(r["advisory"] == "STOP" for r in images))
    free = sum(r["occupied_any"] == "0" for r in images)
    over = sum(int(r["over_caution"]) for r in images)
    check("TestFree", free)
    check("TestOver", over)
    unsupported = sum(int(r["severe_unsupported"]) for r in images)
    check("TestUnsupported", unsupported)
    check("TestUnsupportedPct", pct(unsupported, severe))
    empty = [r for r in images if r["n_pred"] == "0"]
    check("TestEmpty", len(empty))
    check("TestFallback", sum(int(r["fallback_fired"]) for r in images))
    check("TestUnsafeNoFB", sum(int(r["unsafe_clear_nofb"]) for r in images))
    matched = [o for o in objects if o["tp"] == "1"]
    path = [o for o in matched if o["relation"] == "PATH"]
    direct = [o for o in matched if o["gt_lane"] == "direct"]
    check("TestPathPrecN", len(path))
    check("TestPathPrec", pct(sum(o["gt_lane"] == "direct" for o in path), len(path)))
    check("TestPathRec", pct(sum(o["relation"] == "PATH" for o in direct), len(direct)))
    check("TestPathOff", sum(o["gt_lane"] == "off" for o in path))
    values = [float(r["corridor_iou"]) for r in images if r["corridor_iou"] not in ("", "nan")]
    check("TestCorridorIoU", sum(values) / len(values))
    values = [float(r["road_iou"]) for r in images if r["road_iou"] not in ("", "nan")]
    check("TestRoadIoU", sum(values) / len(values))
    for tod, tag in (("night", "Night"), ("daytime", "Day")):
        sel = [r for r in images if r["timeofday"] == tod]
        keys = {r["key"] for r in sel}
        check(f"Test{tag}RecRel", pct(sum(int(r["n_tp_relevant"]) for r in sel), sum(int(r["n_gt_relevant"]) for r in sel)))
        objs = [o for o in objects if o["key"] in keys]
        check(f"Test{tag}StablePct", pct(sum(o["state"] == "STABLE" for o in objs), len(objs)))
        values = [float(r["road_iou"]) for r in sel if r["road_iou"] not in ("", "nan")]
        check(f"Test{tag}RoadIoU", sum(values) / len(values))
    # Truncation by state (matched objects: the flag comes from the annotation).
    for state, tag in (("CHECK", "Check"), ("STABLE", "Stable")):
        sel = [o for o in matched if o["state"] == state]
        check(f"TestTrunc{tag}", pct(sum(o["gt_truncated"] == "1" for o in sel), len(sel)))
    full = {r["key"]: r for r in images}

    def variant(name):
        return {r["key"]: r for r in rows(f"test_{name}", "images")}

    def changed(other_rows, column="advisory"):
        return sum(full[k]["advisory"] != other_rows[k][column] for k in full)

    gates = variant("no_gates")
    check("TestNoGatesStop", sum(r["advisory"] == "STOP" for r in gates.values()))
    check("TestNoGatesOver", sum(int(r["over_caution"]) for r in gates.values()))
    check("TestNoGatesUnsupported", sum(int(r["severe_unsupported"]) for r in gates.values()))
    check("TestNoSQClear", sum(r["advisory"] == "CLEAR" for r in variant("no_scene_quality").values()))
    single = variant("single_region")
    check("TestSingleChanged", changed(single))
    objs = rows("test_single_region", "objects")
    check("TestSingleStablePct", pct(sum(o["state"] == "STABLE" for o in objs), len(objs)))
    check("TestNoDepthChanged", changed(variant("no_depth")))
    objs = rows("test_no_depth", "objects")
    check("TestNoDepthStablePct", pct(sum(o["state"] == "STABLE" for o in objs), len(objs)))
    hyb = variant("no_hybridnets")
    check("TestNoHybChanged", changed(hyb))
    check("TestNoHybUnsafe", sum(int(r["unsafe_clear"]) for r in hyb.values()))
    check("TestNoHybOverPct", pct(sum(int(r["over_caution"]) for r in hyb.values()), free))
    exp = variant("corridor_road")
    check("TestExpOver", sum(int(r["over_caution"]) for r in exp.values()))
    check("TestExpUnsafe", sum(int(r["unsafe_clear"]) for r in exp.values()))
    objs = [o for o in rows("test_corridor_road", "objects") if o["tp"] == "1"]
    path = [o for o in objs if o["relation"] == "PATH"]
    direct = [o for o in objs if o["gt_lane"] == "direct"]
    check("TestExpPathPrec", pct(sum(o["gt_lane"] == "direct" for o in path), len(path)))
    check("TestExpPathRec", pct(sum(o["relation"] == "PATH" for o in direct), len(direct)))

# Operating points (held-out val_rest and train) and the dev curve.
for split, prefix in (("valop", "OpValop"), ("train", "OpTrain")):
    base = rows(f"{split}_full", "images")
    chosen = rows(f"{split}_risk_scale_1.4", "images")
    occupied = sum(int(r["occupied"]) for r in base)
    free = sum(r["occupied_any"] == "0" for r in base)
    check(f"{prefix}Images", len(base))
    check(f"{prefix}BaseUnsafePct", pct(sum(int(r["unsafe_clear"]) for r in base), occupied))
    check(f"{prefix}ChosenUnsafePct", pct(sum(int(r["unsafe_clear"]) for r in chosen), occupied))
    check(f"{prefix}BaseOverPct", pct(sum(int(r["over_caution"]) for r in base), free))
    check(f"{prefix}ChosenOverPct", pct(sum(int(r["over_caution"]) for r in chosen), free))
    check(f"{prefix}ChosenUnsafe", sum(int(r["unsafe_clear"]) for r in chosen))
dev = rows("dev_full", "images")
check("DevImages", len(dev))
check("DevOccupied", sum(int(r["occupied"]) for r in dev))
free = sum(r["occupied_any"] == "0" for r in dev)
check("OpDevBaseOverPct", pct(sum(int(r["over_caution"]) for r in dev), free))
safe_scale = 1.0
for scale in ("1.1", "1.2", "1.4", "1.6", "1.8", "2.0", "2.5"):
    run = rows(f"dev_risk_scale_{scale}", "images")
    if sum(int(r["unsafe_clear"]) for r in run) > 0:
        break
    safe_scale = float(scale)
    safe_over = pct(sum(int(r["over_caution"]) for r in run), free)
check("OpDevSafeScale", safe_scale)
check("OpDevSafeOverPct", safe_over)
last = rows("dev_risk_scale_2.5", "images")
check("OpDevMaxOverPct", pct(sum(int(r["over_caution"]) for r in last), free))
check("OpDevMaxUnsafePct", pct(sum(int(r["unsafe_clear"]) for r in last), sum(int(r["occupied"]) for r in dev)))

# Train sentences.
train = rows("train_full", "images")
check("TrainImages", len(train))
check("TrainOccupied", sum(int(r["occupied"]) for r in train))
check("TrainUnsafe", sum(int(r["unsafe_clear"]) for r in train))
free = sum(r["occupied_any"] == "0" for r in train)
check("TrainOverPct", pct(sum(int(r["over_caution"]) for r in train), free))
values = [float(r["road_iou"]) for r in train if r["road_iou"] not in ("", "nan")]
check("TrainRoadIoU", sum(values) / len(values))
hyb = rows("train_no_hybridnets", "images")
check("TrainNoHybUnsafe", sum(int(r["unsafe_clear"]) for r in hyb))
check("TrainNoHybOverPct", pct(sum(int(r["over_caution"]) for r in hyb), free))
for split, prefix, data in (("test", "Test", rows("test_full", "images")), ("train", "Train", train)):
    for weather, wtag in (("clear", "Clear"), ("rainy", "Rainy"), ("snowy", "Snowy"), ("foggy", "Foggy")):
        sel = [r for r in data if r["weather"] == weather]
        check(f"{prefix}W{wtag}Images", len(sel))
        if weather == "foggy":
            continue
        for tod, ttag in (("daytime", "Day"), ("night", "Night")):
            part = [r for r in sel if r["timeofday"] == tod]
            check(f"{prefix}W{wtag}{ttag}RecRel",
                  pct(sum(int(r["n_tp_relevant"]) for r in part), sum(int(r["n_gt_relevant"]) for r in part)))
        if weather in ("clear", "rainy"):
            check(f"{prefix}W{wtag}NightPct", pct(sum(r["timeofday"] == "night" for r in sel), len(sel)))

# Unsafe-CLEAR breakdown and the night-only replacements.
for split, prefix in (("test", "Test"), ("train", "Train")):
    unsafe = {r["key"]: r for r in rows(f"{split}_full", "images") if r["unsafe_clear"] == "1"}
    found = {}
    for o in rows(f"{split}_full", "objects"):
        if o["key"] in unsafe and o["tp"] == "1" and o["gt_lane"] == "direct" and o["gt_relevant"] == "1":
            found.setdefault(o["key"], []).append(o["relation"])
    check(f"{prefix}UnsafeDetected", len(found))
    check(f"{prefix}UnsafeDetectedSide", sum(set(v) == {"SIDE"} for v in found.values()))
    check(f"{prefix}UnsafeMissed", len(unsafe) - len(found))
    check(f"{prefix}UnsafeMissedNight", sum(r["timeofday"] == "night" for k, r in unsafe.items() if k not in found))
full = {r["key"]: r for r in rows("test_full", "images")}
for name, tag in (("no_depth", "NoDepth"), ("no_hybridnets", "NoHyb")):
    other = {r["key"]: r for r in rows(f"test_{name}", "images")}
    for tod, ttag in (("daytime", "Day"), ("night", "Night")):
        keys = [k for k, r in full.items() if r["timeofday"] == tod]
        check(f"Test{tag}{ttag}ChangedPct", pct(sum(full[k]["advisory"] != other[k]["advisory"] for k in keys), len(keys)))

print(f"\n{sum(results)} of {len(results)} checks pass" + ("" if all(results) else "  <-- FAILURES ABOVE"))
