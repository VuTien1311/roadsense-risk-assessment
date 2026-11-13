"""Download the three official pretrained checkpoints and verify them.

Nothing is hosted by this repository: every file comes from its authors' own release. The URL,
the size and the SHA-256 of each checkpoint are in ``weights_manifest.json``; a download is kept
only if it matches, and a file that is already present and correct is not downloaded again.

    python scripts/download_weights.py           # the three models of the paper
    python scripts/download_weights.py --list    # show sources, sizes and licenses only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problem_with(path, entry):
    """None when the file is the checkpoint of the manifest, else what is wrong with it."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if entry.get("bytes") and size != entry["bytes"]:
        return f"size {size} instead of {entry['bytes']} bytes"
    if entry.get("sha256") and sha256_of(path).lower() != entry["sha256"].lower():
        return "SHA-256 differs from the manifest"
    return None


def fetch(url, destination, expected_bytes=None, report=print):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RoadScene-FailAware weight downloader"})
    done, step = 0, 0
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if expected_bytes and done * 10 // expected_bytes > step:
                step = done * 10 // expected_bytes
                report(f"    {min(step * 10, 100):3d} %  ({done / 1e6:.0f} MB)")
    return partial


def download(entry, root, report=print):
    """Returns True when the checkpoint is in place and verified."""
    target = Path(root) / entry["path"]
    problem = problem_with(target, entry)
    if problem is None:
        report(f"OK       {entry['path']} (already present, verified)")
        return True
    if problem != "missing":
        report(f"REFUSED  {entry['path']}: an existing file is wrong ({problem}). Move it away and run again.")
        return False
    if not entry.get("url"):
        report(f"MISSING  {entry['path']}: no download URL in the manifest; see docs/WEIGHTS.md")
        return False
    report(f"GET      {entry['path']}  <-  {entry['url']}")
    try:
        partial = fetch(entry["url"], target, entry.get("bytes"), report)
    except (OSError, ValueError) as error:
        report(f"FAILED   {entry['path']}: {error}")
        return False
    problem = problem_with(partial, entry)
    if problem is not None:
        partial.unlink(missing_ok=True)
        report(f"FAILED   {entry['path']}: the downloaded file was rejected ({problem})")
        return False
    os.replace(partial, target)
    report(f"OK       {entry['path']} (downloaded, verified)")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="Show sources, sizes and licenses; download nothing.")
    parser.add_argument("--root", default=str(ROOT), help="Repository folder that receives the checkpoints.")
    parser.add_argument("--manifest", default=str(ROOT / "weights_manifest.json"))
    args = parser.parse_args(argv)
    models = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["models"]

    print("Pretrained checkpoints (from their authors' releases; not redistributed by this repository):")
    for entry in models:
        size = f"{entry['bytes'] / 1e6:.0f} MB" if entry.get("bytes") else "size unknown"
        print(f"  {entry['path']}  |  {entry.get('model', '')}  |  {size}  |  license: {entry.get('license', 'see source')}")
    print("Each model keeps its own license; downloading it means accepting those terms.")
    if args.list:
        return 0
    results = [download(entry, args.root) for entry in models]
    if all(results):
        print("All checkpoints are in place. Next: python run_paper.py --check")
        return 0
    print("Some checkpoints are missing; see the messages above and docs/WEIGHTS.md.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
