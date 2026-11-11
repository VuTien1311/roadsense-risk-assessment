# Camera-ready evaluation (RoadScene-FailAware)

Everything in the camera-ready paper is regenerated from this folder. No number
in the manuscript is typed by hand: `paper_numbers.py` writes them as LaTeX
macros from `outputs/camera_ready/results.json`.

## Data

* BDD100K `100k/val` images (10,000) under `../data/images/100k/val`.
* BDD100K val annotations (detection boxes with `occluded` / `truncated`,
  `timeofday` / `weather` / `scene`, drivable maps) from the FiftyOne export of
  the official labels: `samples.json` (SHA-256
  `1e8853904e2926c434557bb90a36b1638ac03ebb77adfc21f04d821a6a998a09`, identical to
  the hash recorded in June 2026) and `fields/drivable/*.png`
  (0 = direct, 1 = alternative, 2 = background), stored in
  `../data/bdd100k_labels_val`.

## Splits (published as ID lists in `sets/`)

| Split | Images | Use | ID-list SHA-256 |
|---|---:|---|---|
| dev | 500 | random val images (seed 20261108); priors, sensitivity, operating-point choice only | see `sets/splits_manifest.json` |
| test | 9,500 | val minus dev; every headline number | idem |
| hard | 100 | deterministic stress subset of test (`static_training/select_hard_bdd100k.py`) | idem |
| standard | 1,000 | random subset of test; perturbation stability | idem |
| val_rest | 8,400 | test minus hard minus standard; held-out operating-point check | idem |
| train | 69,863 | BDD100K train images with det_20 labels; supplementary only (HybridNets was trained on them) | `sets/train_ids.txt` |

## Where each result was computed

Same code (git commit recorded in every Kaggle manifest) and the same pinned
versions (torch 2.7.1+cu118, CUDA 11.8, cuDNN 9.1, ultralytics 8.4.66, numpy
2.4.6, OpenCV 4.13.0.92; xformers absent) on two GPUs:

| Work | GPU |
|---|---|
| hard, dev, standard (incl. all replays), first 5,130 val_rest images | RTX 3050 6 GB laptop |
| remaining 3,270 val_rest images (13 variants, online) | Kaggle T4 (x2) |
| clean 1k subset rerun + 6 in-memory perturbations of it | Kaggle T4 |
| latency benchmark (120 images, alone on one GPU) | Kaggle T4 (separate from the RTX 3050 latency) |
| train split (69,863 images, 9 variants) and val_rest operating points | Kaggle T4 |

Perturbed images are always compared with clean runs on the same GPU. The
RTX 3050 vs T4 agreement is measured on the 1k subset and on the val_rest images
processed locally (`results.json["crosshw"]`). Latency rows of the two GPUs are
never mixed.

## Kaggle runs

The driver scripts of these runs (`kaggle/`, `chain*.sh`) are tied to the authors' accounts and
machines and are not part of this public release; this section documents how the runs were made.

All datasets and kernels are private. The train split ran on two of the authors'
Kaggle accounts; the second one had read-only access to the four input datasets
and nothing else.

1. `build_bundle.py` / `build_train_bundle.py`: code, YOLO weights, images and a
   manifest with SHA-256 of every weight and image (DA V2 and HybridNets weights
   come from an existing private dataset and are hash-checked).
2. `make_kernel.py` + `kaggle kernels push --accelerator NvidiaTeslaT4`:
   `run_kaggle.py` installs the pinned versions, refuses to run on any version or
   hash mismatch, shards the work over both GPUs, stops cleanly before the session
   limit (`harness.py --stop-at`) and zips each job as soon as it ends.
3. `merge.py --src <downloaded out/>` (check) then `--apply`: imports results
   without overwriting any local report; image paths are rewritten to local paths.
4. Train split: the official BDD100K server no longer answers, so `verify_train.py`
   (CPU kernel) accepts third-party copies only because, converted by the same
   code, their val labels equal our val ground truth for all 10,000 images, their
   val drivable masks are pixel-identical, the images match the local SHA-256
   values, and the two train-image copies are identical. `train_driver.py` pushes
   the GPU sessions (resumable across sessions). Keys that no session finished are
   given to new sessions as explicit lists, also on the second account;
   `team_plan.py` reconstructs exactly which keys each session finished from its
   status file. The second account's archives are checked against that plan and
   copied into a private dataset. `eval_train.py` scores all archives in one CPU
   kernel (as many gt_eval runs in parallel as its memory allows), and the driver
   imports the CSVs only if every train key was evaluated in every variant.

## Pipeline

1. `harness.py infer` runs the unmodified `test3.ADASPipeline` and caches the
   three model outputs per image (`outputs/camera_ready/cache`). Its reports are
   byte-identical to `run_paper.py`.
2. `harness.py replay --variant V` re-runs only the fusion stage from the cache.
   `legacy_default`, `risk_default` and `sq_default` re-implement the patched
   functions with the published constants and reproduce the original reports
   exactly (100/100 identical), which validates every ablation built on them.
3. `gt_eval.py` scores a run against the annotations (definitions in its docstring,
   fixed before any result was inspected).
4. `results.py` aggregates metrics with Wilson / image-cluster bootstrap CIs and
   exact McNemar tests; `make_tables.py`, `figures.py` and `paper_numbers.py`
   produce the paper tables, figures and macros.

```powershell
# from RoadScene-FailAware/, with the project's Python environment
python camera_ready/sets.py
python camera_ready/harness.py infer --image-list outputs/camera_ready/sets/hard.txt --out outputs/camera_ready/runs/hard_full
python camera_ready/harness.py infer --image-list outputs/camera_ready/sets/dev.txt --out outputs/camera_ready/runs/dev_full --no-vis
python camera_ready/harness.py infer --image-list outputs/camera_ready/sets/standard.txt --out outputs/camera_ready/runs/standard_full --no-vis
python camera_ready/perturb.py --image-list outputs/camera_ready/sets/hard.txt
python camera_ready/run_all.py gpu priors replays eval summary tables
python camera_ready/paper_numbers.py
```

## Variants

| Name | Meaning |
|---|---|
| `B0` | YOLO with dataset-average road and depth priors, no scene quality, no gates (fallback removed offline) |
| `B1` | B0 + HybridNets corridor |
| `B2` | + Depth Anything V2, centre region only |
| `B3` | + multi-region reliability and scene quality (no gates, no fallback) |
| `full` | published system |
| `no_*`, `single_region` | leave-one-out |
| `sens:*`, `risk:*`, `sq:*` | +/-20 % constant changes on the dev split |
| `corridor_road` | exploratory only: corridor intersected with the dilated HybridNets drivable mask |
| `swap_yolo11s`, `swap_yolo11m` | same fusion layer with smaller YOLO11 detectors; computed but not reported (the paper uses YOLO11x-seg only, as in the submitted version) |

Perturbations (`perturb.py`): gamma 1.5 / 0.67, contrast x0.7, Gaussian blur
sigma 1.5, Gaussian noise sigma 8 (seeded), JPEG quality 35. Geometry is never changed.
