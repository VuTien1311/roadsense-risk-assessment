"""How often does the ego corridor actually follow HybridNets lane pixels?

Re-traces the row loop of ``ADASPipeline.get_ego_corridor`` on cached HybridNets
masks and counts the rows in which both lane boundaries were found and accepted
(``lane-guided`` rows); in all other rows the corridor keeps the default
perspective trapezoid. Writes outputs/camera_ready/corridor_stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "camera_ready"
SETS = Path(__file__).resolve().parent / "sets"


def lane_guided_fraction(road_mask, lane_mask):
    height, width = road_mask.shape
    horizon = int(height * 0.35)
    lane_guidance = cv2.dilate(lane_mask.astype(np.uint8), np.ones((7, 5), np.uint8), iterations=1)
    previous_center = width / 2
    guided = 0
    rows = 0
    for y in range(horizon, height):
        rows += 1
        progress = (y - horizon) / max(height - horizon - 1, 1)
        expected_half_width = width * (0.04 + 0.19 * progress)
        band_y1 = max(horizon, y - 2)
        band_y2 = min(height, y + 3)
        lane_x = np.where(lane_guidance[band_y1:band_y2].any(axis=0))[0]
        left = lane_x[lane_x < previous_center - 4]
        right = lane_x[lane_x > previous_center + 4]
        left_x = int(left.max()) if left.size else None
        right_x = int(right.min()) if right.size else None
        center = previous_center
        if left_x is not None and right_x is not None:
            measured_width = right_x - left_x
            candidate_center = (left_x + right_x) / 2
            if (
                expected_half_width * 0.8 <= measured_width <= expected_half_width * 3.2
                and abs(candidate_center - previous_center) <= expected_half_width * 0.35
            ):
                center = candidate_center
                guided += 1
        max_shift = width * (0.003 + 0.006 * progress)
        center = np.clip(center, previous_center - max_shift, previous_center + max_shift)
        previous_center = center
    return guided / max(rows, 1)


def main():
    stats = {}
    for split in ("hard", "standard", "test"):
        ids = [line.strip() for line in (SETS / f"{split}_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        fractions = []
        for image_id in ids:
            path = OUT / "cache" / "hybridnets" / f"{image_id}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            fractions.append(lane_guided_fraction(data["road"], data["lane"]))
        if fractions:
            values = np.asarray(fractions)
            stats[split] = {
                "images": int(len(values)),
                "mean_lane_guided_rows": float(values.mean()),
                "images_mostly_trapezoid": float(np.mean(values < 0.25)),
                "images_no_lane_rows": float(np.mean(values == 0.0)),
            }
    (OUT / "corridor_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
