"""Depth-reliability validation on KITTI 2D/3D object labels (optional).

KITTI object labels give each object's 3D location in camera coordinates, so the
true depth z (metres along the optical axis) is known. The relative depth index
u of the pipeline is only ordinal, so the test is ordinal too:

* pairwise ordinal accuracy: for every pair of matched objects in one image whose
  true depths differ by at least 1 m, is sign(u_i - u_j) == sign(z_i - z_j)?
  Reported for pairs where both estimates are STABLE vs. pairs with at least one
  non-STABLE estimate, with an image-level cluster bootstrap CI.
* per-image Spearman correlation between u and z (images with >= 3 matches).

Usage (after running the pipeline on KITTI training images, e.g.
``harness.py infer --image-list outputs/camera_ready/sets/kitti.txt --out
outputs/camera_ready/runs/kitti_full --no-vis --no-depth-cache``):

    python camera_ready/kitti_eval.py --kitti D:/data/kitti --run outputs/camera_ready/runs/kitti_full

``--make-list`` writes the image list for the harness first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gt_eval import box_iou, load_reports  # noqa: E402
from stats import cluster_bootstrap_ratio  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "outputs" / "camera_ready"
KITTI_CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"}


def read_labels(path):
    objects = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] not in KITTI_CLASSES:
            continue
        x1, y1, x2, y2 = map(float, parts[4:8])
        x, y, z = map(float, parts[11:14])
        objects.append({
            "label": parts[0],
            "truncated": float(parts[1]),
            "occluded": int(float(parts[2])),
            "box": [x1, y1, x2, y2],
            "z": z,
        })
    return objects


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(kitti_root, run_dir, min_gap=1.0):
    reports = load_reports(run_dir)
    label_dir = Path(kitti_root) / "training" / "label_2"
    stable_num, stable_den, mixed_num, mixed_den = [], [], [], []
    rhos = []
    matched_total = 0
    for key, report in sorted(reports.items()):
        label_path = label_dir / f"{key}.txt"
        if not label_path.exists():
            continue
        gts = read_labels(label_path)
        dets = [d for d in report["detections"] if d.get("original_inverse_depth_unit") is not None]
        used, pairs = set(), []
        for det in sorted(dets, key=lambda d: -d["confidence"]):
            best, best_iou = None, 0.5
            for j, gt in enumerate(gts):
                if j in used:
                    continue
                value = box_iou([float(v) for v in det["box_xyxy"]], gt["box"])
                if value >= best_iou:
                    best, best_iou = j, value
            if best is not None:
                used.add(best)
                pairs.append((det["original_inverse_depth_unit"], gts[best]["z"], det["original_unit_status"] == "STABLE"))
        matched_total += len(pairs)
        s_ok = s_n = m_ok = m_n = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                ui, zi, si = pairs[i]
                uj, zj, sj = pairs[j]
                if abs(zi - zj) < min_gap or ui == uj:
                    continue
                correct = (ui - uj) * (zi - zj) > 0
                if si and sj:
                    s_ok += correct
                    s_n += 1
                else:
                    m_ok += correct
                    m_n += 1
        stable_num.append(s_ok)
        stable_den.append(s_n)
        mixed_num.append(m_ok)
        mixed_den.append(m_n)
        if len(pairs) >= 3:
            rhos.append(spearman([p[0] for p in pairs], [p[1] for p in pairs]))
    stable = cluster_bootstrap_ratio(stable_num, stable_den)
    mixed = cluster_bootstrap_ratio(mixed_num, mixed_den)
    rhos = [r for r in rhos if r == r]
    return {
        "images": len(stable_num),
        "matched_objects": matched_total,
        "pairs_stable": int(sum(stable_den)),
        "pairs_non_stable": int(sum(mixed_den)),
        "ordinal_accuracy_stable": stable,
        "ordinal_accuracy_non_stable": mixed,
        "spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "spearman_images": len(rhos),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti", required=True, help="KITTI object root containing training/image_2 and training/label_2")
    parser.add_argument("--run", default=str(OUT / "runs" / "kitti_full"))
    parser.add_argument("--make-list", action="store_true")
    args = parser.parse_args()
    if args.make_list:
        images = sorted((Path(args.kitti) / "training" / "image_2").glob("*.png"))
        listing = OUT / "sets" / "kitti.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8", newline="\n")
        print(f"[KITTI] {len(images)} images -> {listing}")
        return
    result = evaluate(args.kitti, args.run)
    (OUT / "kitti_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
