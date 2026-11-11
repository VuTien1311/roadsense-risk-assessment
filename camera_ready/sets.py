"""Build the camera-ready image splits from BDD100K val.

* hard     : the deterministic 100-image hard set from select_hard_bdd100k.py
* dev      : 500 random val images (seeded), disjoint from hard; used only for
             component priors and threshold-sensitivity analysis
* standard : 1000 random val images (seeded), disjoint from hard and dev; used to
             check that conclusions hold outside the hand-crafted hard set

Publishable ID lists (no images) are written to camera_ready/sets/ together with
their SHA-256 digests; absolute-path lists for local runs go to outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261108


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", default=str((ROOT.parent / "data" / "images" / "100k" / "val").resolve()))
    parser.add_argument("--hard-manifest", default=str(ROOT / "static_training" / "hard_bdd100k" / "manifest.json"))
    parser.add_argument("--dev-count", type=int, default=500)
    parser.add_argument("--standard-count", type=int, default=1000)
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    all_ids = sorted(path.stem for path in val_dir.glob("*.jpg"))
    hard_records = json.loads(Path(args.hard_manifest).read_text(encoding="utf-8"))
    hard_ids = [Path(item["path"]).stem for item in hard_records]
    hard_categories = {Path(item["path"]).stem: item["category"] for item in hard_records}

    remaining = [image_id for image_id in all_ids if image_id not in set(hard_ids)]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(remaining))
    dev_ids = sorted(remaining[i] for i in order[: args.dev_count])
    standard_ids = sorted(
        remaining[i] for i in order[args.dev_count : args.dev_count + args.standard_count]
    )

    publish_dir = ROOT / "camera_ready" / "sets"
    local_dir = ROOT / "outputs" / "camera_ready" / "sets"
    publish_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "BDD100K 100k/val",
        "val_images": len(all_ids),
        "seed": SEED,
        "splits": {},
    }
    used = set(hard_ids) | set(dev_ids) | set(standard_ids)
    rest_ids = [image_id for image_id in all_ids if image_id not in used]
    dev_set = set(dev_ids)
    test_ids = [image_id for image_id in all_ids if image_id not in dev_set]
    splits = (
        ("hard", hard_ids),
        ("dev", dev_ids),
        ("standard", standard_ids),
        ("test", test_ids),  # val minus dev: every reported test number (9,500 images)
        ("val", all_ids),  # the complete BDD100K validation split
        ("val_rest", rest_ids),  # val images not already run in another split
    )
    for name, ids in splits:
        id_text = "\n".join(ids) + "\n"
        (publish_dir / f"{name}_ids.txt").write_text(id_text, encoding="utf-8", newline="\n")
        paths = "\n".join(str((val_dir / f"{image_id}.jpg").resolve()) for image_id in ids) + "\n"
        (local_dir / f"{name}.txt").write_text(paths, encoding="utf-8", newline="\n")
        summary["splits"][name] = {"count": len(ids), "ids_sha256": sha256_text(id_text)}
    (publish_dir / "hard_categories.json").write_text(
        json.dumps(hard_categories, indent=2), encoding="utf-8"
    )
    assert not set(hard_ids) & set(dev_ids)
    assert not set(hard_ids) & set(standard_ids)
    assert not set(dev_ids) & set(standard_ids)
    (publish_dir / "splits_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
