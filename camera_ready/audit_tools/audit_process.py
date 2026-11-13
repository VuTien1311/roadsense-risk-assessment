"""Process checks: splits, hashes, report counts, evaluation coverage. No project code is imported.

Usage: python audit_process.py <repo root>
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
SETS = ROOT / "camera_ready" / "sets"
OUT = ROOT / "outputs" / "camera_ready"
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (OUT / "paper" / "numbers.tex").read_text(encoding="utf-8")))
ok_all = []


def say(ok, text):
    ok_all.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + text)


def ids(name):
    return [line.strip() for line in (SETS / f"{name}_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


sets = {name: ids(name) for name in ("val", "dev", "test", "hard", "standard", "val_rest", "train")}
size = {"val": 10000, "dev": 500, "test": 9500, "hard": 100, "standard": 1000, "val_rest": 8400, "train": 69863}
for name, items in sets.items():
    say(len(items) == size[name] and len(set(items)) == len(items), f"{name}: {len(items)} unique ids (expected {size[name]})")
S = {k: set(v) for k, v in sets.items()}
say(S["dev"] | S["test"] == S["val"] and not (S["dev"] & S["test"]), "dev and test are disjoint and together equal val")
say(S["hard"] <= S["test"] and S["standard"] <= S["test"], "hard and standard are subsets of test")
say(not (S["hard"] & S["standard"]), f"hard and standard are disjoint (overlap {len(S['hard'] & S['standard'])})")
say(S["val_rest"] == S["test"] - S["hard"] - S["standard"], "val_rest = test - hard - standard")
say(not (S["train"] & S["val"]), "train and val share no image id")

manifest = json.loads((SETS / "splits_manifest.json").read_text(encoding="utf-8"))
print("manifest keys:", list(manifest)[:12])
for name, macro in (("test", "TestIDsHash"), ("hard", "HardIDsHash")):
    digest = hashlib.sha256((SETS / f"{name}_ids.txt").read_bytes()).hexdigest()
    say(digest.startswith(MACROS[macro]), f"{name}_ids.txt sha256 {digest[:16]} starts with the paper's prefix {MACROS[macro]}")
    recorded = json.dumps(manifest)
    say(digest in recorded, f"{name}_ids.txt sha256 is the one recorded in splits_manifest.json")
say(str(manifest.get("seed", manifest.get("dev_seed", ""))) == MACROS["SplitSeed"] or MACROS["SplitSeed"] in json.dumps(manifest),
    f"split seed in the paper ({MACROS['SplitSeed']}) is in the manifest")

# Every evaluated run covers exactly its split, once.
expect = {"test": S["test"], "hard": S["hard"], "standard": S["standard"], "dev": S["dev"], "train": S["train"],
          "valop": S["val_rest"]}
bad = 0
count = 0
for folder in sorted((OUT / "eval").iterdir()):
    split = folder.name.split("_")[0]
    if split not in expect:
        continue
    with open(folder / "images.csv", encoding="utf-8", newline="") as handle:
        keys = [row["key"] for row in csv.DictReader(handle)]
    count += 1
    if len(keys) != len(set(keys)) or set(keys) != expect[split]:
        bad += 1
        print(f"   MISMATCH {folder.name}: {len(keys)} rows, {len(set(keys))} unique, expected {len(expect[split])}")
say(bad == 0, f"{count} evaluated runs each cover exactly their split, one row per image")

# Reports on disk for the headline runs.
for run, name in (("test_full", "test"), ("hard_full", "hard"), ("dev_full", "dev"), ("standard_full", "standard")):
    reports = {p.stem.replace("upgraded_", "", 1) for p in (OUT / "runs" / run).glob("upgraded_*.json")}
    say(reports == S[name], f"runs/{run}: {len(reports)} reports = the {name} id list")
print(f"\n{sum(ok_all)} of {len(ok_all)} process checks pass" + ("" if all(ok_all) else "  <-- FAILURES ABOVE"))
