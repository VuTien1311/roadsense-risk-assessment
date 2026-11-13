"""Write a Kaggle kernel folder (script + kernel-metadata.json) for run_kaggle.py.

    python camera_ready/kaggle/make_kernel.py --slug roadscene-cr-smoke --smoke 2
    python camera_ready/kaggle/make_kernel.py --slug roadscene-cr-run --smoke 0
    python camera_ready/kaggle/make_kernel.py --slug roadscene-cr-train-1 --mode train --part 1/2 --operating
    python camera_ready/kaggle/make_kernel.py --slug roadscene-cr-train-3 --mode train --part 1/1 \
        --after roadscene-cr-train-1 roadscene-cr-train-2
    python camera_ready/kaggle/make_kernel.py --slug roadscene-cr-team-1 --mode train --owner <team-account> \
        --assign keys.txt --input-dataset <team-account>/roadscene-cr-team-input
    kaggle kernels push -p <printed folder> --accelerator NvidiaTeslaT4

A kernel of another account (the team account, owner <team-account>) reads the main account's datasets
(shared read-only), its own small dataset with verify_report.json and train_keys.txt (it cannot
read the main account's kernel outputs), and runs exactly the keys of --assign.

Every kernel is private and uses GPU T4 x2 and Internet (pip installs pinned versions).
Val mode mounts roadscene-cr-bundle and edge-adas-teachers-code. Train mode also mounts
roadscene-cr-train-bundle, the user's bdd100k-edge-adas-data (images) and the output of the
verification kernel roadscene-cr-verify, plus earlier train sessions given with --after
(their finished images are skipped).
"""
from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
OWNER = os.environ.get("KAGGLE_OWNER", "your-kaggle-user")
VAL_DATASETS = [f"{OWNER}/roadscene-cr-bundle", f"{OWNER}/edge-adas-teachers-code"]
TRAIN_DATASETS = VAL_DATASETS + [f"{OWNER}/roadscene-cr-train-bundle", f"{OWNER}/bdd100k-edge-adas-data"]


def set_constant(source, name, value):
    source, count = re.subn(rf"^{name} = .*?(  #.*)?$", lambda m: f"{name} = {value}{m.group(1) or ''}", source,
                            count=1, flags=re.M)
    assert count == 1, f"constant {name} not found in run_kaggle.py"
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--budget-h", type=float, default=11.0)
    parser.add_argument("--mode", choices=("val", "train"), default="val")
    parser.add_argument("--part", default="1/1", help="train mode: part i/n of the remaining train images")
    parser.add_argument("--base", default="1/1",
                        help="train mode: restrict to part i/n of ALL train keys (the half of an earlier --part i/n session)")
    parser.add_argument("--operating", action="store_true", help="train mode: also run the val_rest operating points")
    parser.add_argument("--after", nargs="*", default=[], help="train mode: earlier train kernel slugs")
    parser.add_argument("--owner", default=OWNER,
                        help="Kaggle account that owns the kernel (the datasets stay the main account's, shared read-only)")
    parser.add_argument("--assign", type=Path,
                        help="train mode: file with exactly the train keys this session runs (replaces --base/--part)")
    parser.add_argument("--input-dataset",
                        help="train mode, other account: dataset with verify_report.json and train_keys.txt "
                             "(used instead of the main account's roadscene-cr-verify kernel output)")
    args = parser.parse_args()
    if args.owner != OWNER and not (args.mode == "train" and args.assign and args.input_dataset and not args.after):
        parser.error("a kernel of another account needs --mode train, --assign and --input-dataset, and no --after")
    source = (HERE / "run_kaggle.py").read_text(encoding="utf-8")
    source = set_constant(source, "SMOKE", str(args.smoke))
    source = set_constant(source, "BUDGET_H", str(args.budget_h))
    source = set_constant(source, "MODE", json.dumps(args.mode))
    source = set_constant(source, "TRAIN_PART", json.dumps(args.part))
    source = set_constant(source, "TRAIN_BASE", json.dumps(args.base))
    source = set_constant(source, "RUN_OPERATING", str(bool(args.operating)))
    if args.assign:
        keys = args.assign.read_text(encoding="utf-8").split()
        assert keys and len(set(keys)) == len(keys), f"empty or repeated keys in {args.assign}"
        source = set_constant(source, "TRAIN_ASSIGN", json.dumps(" ".join(keys)))
    folder = WORKSPACE / "tmp" / "kaggle_kernels" / args.slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run_kaggle.py").write_text(source, encoding="utf-8", newline="\n")
    title = " ".join(word.upper() if word == "cr" else word.capitalize() for word in args.slug.split("-"))
    train = args.mode == "train"
    if args.input_dataset:
        datasets, kernels = TRAIN_DATASETS + [args.input_dataset], [f"{args.owner}/{s}" for s in args.after]
    elif train:
        datasets, kernels = TRAIN_DATASETS, [f"{OWNER}/roadscene-cr-verify"] + [f"{OWNER}/{s}" for s in args.after]
    else:
        datasets, kernels = VAL_DATASETS, []
    metadata = {
        "id": f"{args.owner}/{args.slug}",
        "title": title,
        "code_file": "run_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": kernels,
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(folder)


if __name__ == "__main__":
    main()
