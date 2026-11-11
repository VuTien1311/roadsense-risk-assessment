"""Dataset-average component priors for the leave-one-component-out ablations.

Computed on the dev split only (never on hard or standard images):

* road_prior.npy  : per-pixel mean of HybridNets road masks, thresholded at 0.5
                    (replaces HybridNets in the ``no_hybridnets`` variant; lanes
                    are left empty, so the ego corridor falls back to the fixed
                    perspective trapezoid in ``get_ego_corridor``)
* depth_prior.npy : per-pixel median of the Depth Anything V2 raw response
                    (replaces the per-image depth in the ``no_depth`` variant)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "camera_ready" / "cache"
OUT = ROOT / "outputs" / "camera_ready" / "priors"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default=str(ROOT / "camera_ready" / "sets" / "dev_ids.txt"))
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args()
    ids = [line.strip() for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = Path(args.cache)
    road_sum = None
    depth_small = []
    shape = None
    for image_id in ids:
        road = np.load(cache / "hybridnets" / f"{image_id}.npz")["road"].astype(np.float64)
        depth = np.load(cache / "depth" / f"{image_id}.npy").astype(np.float32)
        if shape is None:
            shape = road.shape
        if road.shape != shape:
            road = cv2.resize(road, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        road_sum = road if road_sum is None else road_sum + road
        depth_small.append(cv2.resize(depth, (shape[1] // 4, shape[0] // 4), interpolation=cv2.INTER_AREA))
    road_prior = (road_sum / len(ids) >= 0.5).astype(np.uint8)
    depth_prior = np.median(np.stack(depth_small), axis=0).astype(np.float32)
    depth_prior = cv2.resize(depth_prior, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "road_prior.npy", road_prior)
    np.save(OUT / "depth_prior.npy", depth_prior)
    print(f"images={len(ids)} shape={shape} road_fraction={road_prior.mean():.3f} "
          f"depth_range={depth_prior.min():.1f}-{depth_prior.max():.1f}")


if __name__ == "__main__":
    main()
