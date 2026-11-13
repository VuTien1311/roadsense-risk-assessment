# Paper

Camera-ready manuscript of "Conservative Road-Scene Risk Assessment via Fail-Aware Multi-Model Fusion"
(ICoCSETI 2026), accepted version.

* `main.pdf`: the six-page paper. `main_extended.pdf`: the same paper with a supplementary appendix
  (all tables, six qualitative cases, train-split check).
* `main.tex` reads every number from `numbers.tex`, which `camera_ready/paper_numbers.py` writes from
  `results/results.json`; `tables/` and `figures/` are written by `camera_ready/make_tables.py` and
  `camera_ready/figures.py`. No number is typed by hand.
* Build: `pdflatex main.tex` twice (IEEEtran). Extended version:
  `pdflatex -jobname=main_extended "\def\Extended{1}\input{main}"`.

The final published version is the one in IEEE Xplore; this copy is provided for personal and
research use.
