"""Assemble the 9,500-image test split (BDD100K val minus dev) per variant.

The test split was processed in three disjoint pieces: the hard set and the
1,000-image random subset (model outputs cached, ablations replayed) and the
remaining 8,400 images (ablations computed online). This script hard-links the
per-image reports of the three pieces into ``runs/test_<variant>`` and checks
that every test image is present exactly once.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "camera_ready" / "runs"
SETS = Path(__file__).resolve().parent / "sets"
PIECES = ("hard", "standard", "val_rest")
VARIANTS = (
    "full",
    "B0",
    "B1",
    "B2",
    "B3",
    "no_hybridnets",
    "no_depth",
    "single_region",
    "no_scene_quality",
    "no_gates",
    "corridor_road",
    "swap_yolo11s",
    "swap_yolo11m",
)


def link(source, target):
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    expected = {line.strip() for line in (SETS / "test_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    summary = {}
    for variant in VARIANTS:
        target = RUNS / f"test_{variant}"
        target.mkdir(parents=True, exist_ok=True)
        found = set()
        for piece in PIECES:
            source = RUNS / f"{piece}_{variant}"
            if not source.exists():
                continue
            for path in source.glob("upgraded_*.json"):
                key = path.stem.replace("upgraded_", "", 1)
                if key in expected and key not in found:
                    found.add(key)
                    link(path, target / path.name)
        summary[variant] = (len(found), len(expected - found))
        print(f"[MERGE] test_{variant}: {len(found)} reports, missing {len(expected - found)}")
    return summary


if __name__ == "__main__":
    main()
