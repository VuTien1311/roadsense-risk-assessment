"""Objective breakdown of the unsafe-CLEAR test images from the evaluation CSVs.

Usage: python unsafe_cases.py <camera_ready output folder> [run]
For every unsafe-CLEAR image: was a relevant direct-lane road user among the accepted detections
(then the fusion scored it below MONITOR), or did the detector miss it?
"""
import csv
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
RUN = sys.argv[2] if len(sys.argv) > 2 else "test_full"


def rows(kind):
    with open(OUT / "eval" / RUN / f"{kind}.csv", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


images = [r for r in rows("images") if r["unsafe_clear"] == "1"]
objects = {}
for o in rows("objects"):
    objects.setdefault(o["key"], []).append(o)
detected = missed = 0
by_relation = {}
night = {"detected": 0, "missed": 0}
for image in images:
    hits = [o for o in objects.get(image["key"], [])
            if o["tp"] == "1" and o["gt_lane"] == "direct" and o["gt_relevant"] == "1"]
    kind = "detected" if hits else "missed"
    detected += bool(hits)
    missed += not hits
    night[kind] += image["timeofday"] == "night"
    for o in hits:
        by_relation[o["relation"]] = by_relation.get(o["relation"], 0) + 1
    print(image["key"], image["timeofday"], image["weather"], f"scene={image['scene_status']}",
          f"n_pred={image['n_pred']} n_gt_relevant={image['n_gt_relevant']}", kind,
          [(o["class"], o["relation"], o["state"], o["advisory"], round(float(o["risk"]), 1)) for o in hits])
print(f"\n{RUN}: {len(images)} unsafe CLEAR images: in-lane relevant road user detected in {detected} "
      f"(night {night['detected']}), missed by the detector in {missed} (night {night['missed']})")
print("path relation of the detected in-lane objects:", by_relation)
