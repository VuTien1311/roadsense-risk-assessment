#!/usr/bin/env python3
"""List what the mounted BDD100K candidate datasets contain (CPU kernel, read only).

For every dataset under /kaggle/input: folders with file counts and example names,
JSON label files (size and the start of the content), zip members, and the pixel
values of one drivable-area mask. Also checks whether the official BDD100K server
answers from Kaggle. Writes /kaggle/working/probe.txt.
"""
import json
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

OUT = Path("/kaggle/working/probe.txt")
lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(str(text))


def summarise(root):
    counts, examples, sizes = defaultdict(int), defaultdict(list), defaultdict(int)
    special = []
    for folder, _dirs, files in os.walk(root):
        rel = os.path.relpath(folder, root)
        for name in files:
            counts[rel] += 1
            path = os.path.join(folder, name)
            try:
                sizes[rel] += os.path.getsize(path)
            except OSError:
                pass
            if len(examples[rel]) < 2:
                examples[rel].append(name)
            if name.endswith((".json", ".zip")) or ("drivable" in path.lower() and name.endswith(".png") and len(special) < 40):
                special.append(path)
    for rel in sorted(counts):
        say(f"  {rel:70s} {counts[rel]:7d} files {sizes[rel] / 2**20:9.1f} MB  e.g. {examples[rel]}")
    return special


def inspect(path):
    try:
        size = os.path.getsize(path)
        if path.endswith(".json"):
            with open(path, "rb") as handle:
                head = handle.read(600).decode("utf-8", "replace").replace("\n", " ")
            say(f"  JSON {path} ({size / 2**20:.1f} MB): {head[:600]}")
        elif path.endswith(".zip"):
            with zipfile.ZipFile(path) as handle:
                names = handle.namelist()
            say(f"  ZIP {path} ({size / 2**20:.1f} MB): {len(names)} members, e.g. {names[:12]}")
        elif path.endswith(".png"):
            import numpy as np
            from PIL import Image

            array = np.array(Image.open(path))
            values, freq = np.unique(array.reshape(-1, array.shape[-1]) if array.ndim == 3 else array, return_counts=True, axis=0)
            say(f"  PNG {path}: shape {array.shape}, values {values.tolist()[:6]}")
    except Exception as error:  # report and continue
        say(f"  could not read {path}: {error!r}")


def main():
    base = Path("/kaggle/input")
    # Datasets are mounted as /kaggle/input/datasets/<owner>/<slug> (older images: /kaggle/input/<slug>).
    roots = sorted(base.glob("datasets/*/*")) if (base / "datasets").is_dir() else sorted(base.glob("*"))
    for root in roots:
        say(f"=== {root}")
        seen_png = 0
        for path in summarise(root):
            if path.endswith(".png"):
                seen_png += 1
                if seen_png > 2:
                    continue
            inspect(path)
    say("=== official server from Kaggle")
    for name in ("bdd100k_det_20_labels_trainval.zip", "bdd100k_drivable_labels_trainval.zip", "100k_images_train.zip"):
        result = subprocess.run(["curl", "-sI", "-L", "--max-time", "30", f"https://dl.cv.ethz.ch/bdd100k/data/{name}"],
                                capture_output=True, text=True)
        status = [l for l in result.stdout.splitlines() if l.upper().startswith(("HTTP/", "CONTENT-LENGTH"))]
        say(f"  {name}: {status or result.stderr.strip()[:200] or 'no answer'}")
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
