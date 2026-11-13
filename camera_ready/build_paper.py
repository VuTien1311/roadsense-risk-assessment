"""Assemble and compile the camera-ready manuscript.

Copies the generated tables and figures next to ``main.tex`` (plus the original
pipeline figure extracted from the submitted PDF), regenerates ``numbers.tex``
and runs pdflatex twice. Reports undefined macros and overfull boxes.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "camera_ready"
PAPER = OUT / "paper"


def main():
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    for path in (OUT / "figures").glob("*.pdf"):
        shutil.copy2(path, PAPER / "figures" / path.name)
    pipeline = OUT / "pdf_extract" / "p3-000.png"
    if pipeline.exists():
        shutil.copy2(pipeline, PAPER / "figures" / "fig_pipeline.png")
    for path in (OUT / "tables").glob("*.tex"):
        shutil.copy2(path, PAPER / "tables" / path.name)
    subprocess.run([sys.executable, str(Path(__file__).with_name("paper_numbers.py"))], check=False)
    jobs = (
        ("main", "main.tex"),  # camera-ready version
        ("main_extended", r"\def\Extended{1}\input{main.tex}"),  # with supplementary appendix
    )
    for jobname, source in jobs:
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", f"-jobname={jobname}", source],
                cwd=PAPER,
                capture_output=True,
                text=True,
                timeout=300,
            )
        log = (PAPER / f"{jobname}.log").read_text(encoding="utf-8", errors="replace")
        pages = re.search(rf"Output written on {jobname}\.pdf \((\d+) pages?", log)
        undefined = sorted(set(re.findall(r"Undefined control sequence\.\s*\nl\.\d+ .*\\(\w+)\s*\n", log)))
        overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
        print(f"[PAPER] {jobname}: rc={result.returncode} pages={pages.group(1) if pages else '?'} "
              f"undefined={undefined} overfull_boxes={len(overfull)} "
              f"max_overfull={max(map(float, overfull)) if overfull else 0:.1f}pt")
        if result.returncode != 0 and not pages:
            print("\n".join(log.splitlines()[-25:]))


if __name__ == "__main__":
    main()
