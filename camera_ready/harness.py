"""Cached experiment runner for the RoadScene-FailAware camera-ready revision.

Both modes drive the unmodified ``test3.ADASPipeline.process_image``:

* ``infer``  loads the three pretrained models, runs the original pipeline and
  caches each model's output per image (HybridNets road/lane masks, Depth
  Anything V2 raw response, filtered YOLO detections with masks). It also
  records per-component GPU latency and peak memory.
* ``replay`` loads no network. It re-runs only the fusion stage from the cache,
  optionally with one ablation variant patched in (see ``variants.py``).

Replaying the ``full`` variant must reproduce the ``infer`` reports exactly;
``check`` verifies this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import test3  # noqa: E402
import torch  # noqa: E402

CONFIG = json.loads((ROOT / "paper_config.json").read_text(encoding="utf-8"))
PIPELINE_SETTINGS = {
    "detection_confidence": float(CONFIG["confidence"]),
    "detection_image_size": int(CONFIG["image_size"]),
    "depth_input_size": int(CONFIG["depth_size"]),
    "enhance_input": CONFIG["enhance_input"],
    "yolo_tta": CONFIG["yolo_tta"],
    "adaptive_conf": bool(CONFIG["adaptive_confidence"]),
    "yolo_imgszs": test3.parse_imgszs(CONFIG["yolo_imgszs"]),
    "tile_inference": CONFIG["tile_inference"],
    "tile_size": 640,
    "tile_overlap": 0.20,
    "precision_mode": CONFIG["precision_mode"],
}
DISTANCE_MODE = CONFIG["distance_mode"]
DEFAULT_CACHE = ROOT / "outputs" / "camera_ready" / "cache"


def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def yolo_tag(weight):
    keys = (
        "detection_confidence",
        "detection_image_size",
        "enhance_input",
        "yolo_tta",
        "adaptive_conf",
        "yolo_imgszs",
        "tile_inference",
        "tile_size",
        "tile_overlap",
        "precision_mode",
    )
    subset = {key: PIPELINE_SETTINGS[key] for key in keys}
    digest = hashlib.sha1(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:8]
    return f"{Path(weight).stem}-{digest}"


def pack_detection(detection):
    packed = {key: value for key, value in detection.items() if key != "mask"}
    mask = detection.get("mask")
    if mask is None:
        packed["mask"] = None
        return packed
    mask = np.asarray(mask)
    binary = bool(np.isin(np.unique(mask), (0, 1)).all())
    if binary:
        packed["mask"] = {
            "kind": "bits",
            "shape": mask.shape,
            "dtype": str(mask.dtype),
            "bits": np.packbits(mask.astype(bool).ravel()),
        }
    else:
        packed["mask"] = {"kind": "raw", "array": mask}
    return packed


def unpack_detection(packed):
    detection = dict(packed)
    mask = packed.get("mask")
    if mask is None:
        detection["mask"] = None
    elif mask["kind"] == "bits":
        size = int(np.prod(mask["shape"]))
        flat = np.unpackbits(mask["bits"])[:size]
        detection["mask"] = flat.reshape(mask["shape"]).astype(mask["dtype"])
    else:
        detection["mask"] = mask["array"]
    return detection


class ComponentCache:
    def __init__(self, root, weight):
        self.root = Path(root)
        self.hybrid_dir = self.root / "hybridnets"
        self.depth_dir = self.root / "depth"
        self.yolo_dir = self.root / "yolo" / yolo_tag(weight)
        for directory in (self.hybrid_dir, self.depth_dir, self.yolo_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def hybrid(self, key):
        return self.hybrid_dir / f"{key}.npz"

    def depth(self, key):
        return self.depth_dir / f"{key}.npy"

    def yolo(self, key):
        return self.yolo_dir / f"{key}.pkl"


class DepthProxy:
    """Stands in for ``pipeline.depth_model`` inside ``process_image``."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def infer_image(self, raw_image, input_size=518):
        pipeline = self.pipeline
        if pipeline.depth_override is not None:
            return pipeline.depth_override(raw_image)
        memo = pipeline.memo.get("depth")
        if memo is not None and memo[0] == pipeline.current_key:
            return memo[1].copy()
        path = pipeline.cache.depth(pipeline.current_key)
        if path.exists():
            raw_depth = np.load(path)
        else:
            if pipeline.real_depth_model is None:
                raise RuntimeError(f"Depth cache miss in replay mode: {path}")
            start = sync_time()
            raw_depth = pipeline.real_depth_model.infer_image(raw_image, input_size)
            pipeline.timing["depth_s"] = sync_time() - start
            if pipeline.save_depth:
                np.save(path, raw_depth)
        pipeline.memo["depth"] = (pipeline.current_key, raw_depth)
        return raw_depth.copy()


class CachedPipeline(test3.ADASPipeline):
    def __init__(self, yolo_weight, cache_root=DEFAULT_CACHE, load_models=True):
        yolo_weight = str(Path(yolo_weight).resolve())
        self.cache = ComponentCache(cache_root, yolo_weight)
        self.load_models = load_models
        self.current_key = None
        self.timing = {}
        self.road_override = None
        self.depth_override = None
        self.save_depth = True
        self.save_yolo = True
        self.perturbation = None
        self.memo = {}
        if load_models:
            super().__init__(yolo_weight, aux_yolo_weight=None, distance_mode=DISTANCE_MODE)
            self.real_depth_model = self.depth_model
        else:
            # Same attributes as ADASPipeline.__init__ without loading networks.
            self.camera_height_m = test3.DEFAULT_CAMERA_HEIGHT_M
            self.camera_height_source = "default"
            self.horizontal_fov_deg = None
            self.horizon_ratio = None
            self.distance_mode = DISTANCE_MODE
            self.detectors = [
                {"role": "primary", "model": None, "classes": [], "names": {}, "weight": yolo_weight}
            ]
            self.last_yolo_filter_info = {}
            self.real_depth_model = None
        self.depth_model = DepthProxy(self)

    def get_hybridnets_masks(self, image):
        if self.road_override is not None:
            return self.road_override(image)
        memo = self.memo.get("hybrid")
        if memo is not None and memo[0] == self.current_key:
            return memo[1].copy(), memo[2].copy()
        path = self.cache.hybrid(self.current_key)
        if path.exists():
            data = np.load(path)
            road_mask, lane_mask = data["road"], data["lane"]
        else:
            if not self.load_models:
                raise RuntimeError(f"HybridNets cache miss in replay mode: {path}")
            start = sync_time()
            road_mask, lane_mask = super().get_hybridnets_masks(image)
            self.timing["hybridnets_s"] = sync_time() - start
            np.savez_compressed(path, road=road_mask, lane=lane_mask)
        self.memo["hybrid"] = (self.current_key, road_mask, lane_mask)
        return road_mask.copy(), lane_mask.copy()

    def detect_yolo_objects(self, *args, **kwargs):
        memo = self.memo.get("yolo")
        memo_key = (self.current_key, str(self.cache.yolo_dir))
        if memo is not None and memo[0] == memo_key:
            payload = memo[1]
        else:
            path = self.cache.yolo(self.current_key)
            if path.exists():
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
            else:
                if not self.load_models:
                    raise RuntimeError(f"YOLO cache miss in replay mode: {path}")
                start = sync_time()
                detections = super().detect_yolo_objects(*args, **kwargs)
                self.timing["yolo_s"] = sync_time() - start
                payload = {
                    "detections": [pack_detection(item) for item in detections],
                    "filter_info": self.last_yolo_filter_info,
                }
                if self.save_yolo:
                    with path.open("wb") as handle:
                        pickle.dump(payload, handle, protocol=4)
            self.memo["yolo"] = (memo_key, payload)
        self.last_yolo_filter_info = payload["filter_info"]
        return [unpack_detection(item) for item in payload["detections"]]

    def process_image(self, image_path, output_dir, **settings):
        stem = Path(image_path).stem
        self.current_key = f"{stem}__{self.perturbation}" if self.perturbation else stem
        self.timing = {}
        if self.load_models and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = sync_time()
        result = super().process_image(image_path, output_dir, **settings)
        self.timing["total_s"] = sync_time() - start
        if self.load_models and torch.cuda.is_available():
            self.timing["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
        return result


class Cv2NoWrite:
    """Proxy for ``test3.cv2`` that skips image writes during replays."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def imwrite(*_args, **_kwargs):
        return True


class Cv2Perturbed:
    """Proxy for ``test3.cv2`` whose ``imread`` returns a perturbed image.

    The perturbation is applied in memory, identically to perturb.py (the JPEG
    case is encoded and decoded at quality 35), so no image files are written.
    """

    def __init__(self, real, name):
        self._real = real
        self._name = name

    def __getattr__(self, name):
        return getattr(self._real, name)

    def imread(self, path, *args):
        from perturb import perturb

        image = self._real.imread(path, *args)
        if image is None:
            return None
        output = perturb(image, self._name, Path(path).stem)
        if self._name == "jpeg":
            ok, buffer = self._real.imencode(".jpg", output, [self._real.IMWRITE_JPEG_QUALITY, 35])
            output = self._real.imdecode(buffer, self._real.IMREAD_COLOR)
        return output


def read_image_list(path):
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(str(Path(line).resolve()) if not Path(line).is_absolute() else line)
    return items


def run(args):
    images = read_image_list(args.image_list)
    if args.limit:
        images = images[: args.limit]
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    load_models = args.mode == "infer"
    if args.mode == "replay" or args.no_vis or args.variants:
        test3.save_visualizations = lambda *a, **k: {}
        test3.cv2 = Cv2NoWrite(test3.cv2)
    if args.perturb:
        test3.cv2 = Cv2Perturbed(test3.cv2, args.perturb)
    pipeline = CachedPipeline(args.yolo_weight, args.cache, load_models=load_models)
    pipeline.save_depth = not args.no_depth_cache
    pipeline.save_yolo = not args.no_yolo_cache
    pipeline.perturbation = args.perturb or None
    import variants

    if args.variant and args.variant != "full":
        variants.apply(args.variant, pipeline)
    online = [v for v in (args.variants.split(";") if args.variants else []) if v]
    prefix = output_dir.name[:-5] if output_dir.name.endswith("_full") else output_dir.name
    online_dirs = {}
    swaps = {}
    for variant in online:
        if variant.startswith("swap:"):
            # Detector swap: same HybridNets/Depth outputs, another YOLO11 checkpoint.
            weight = str((ROOT / variant[5:]).resolve())
            primary = list(pipeline.detectors)
            pipeline._load_yolo_detector("primary", weight)
            swaps[variant] = {
                "detectors": [pipeline.detectors[-1]],
                "yolo_dir": pipeline.cache.root / "yolo" / yolo_tag(weight),
            }
            swaps[variant]["yolo_dir"].mkdir(parents=True, exist_ok=True)
            pipeline.detectors = primary
            stem = Path(weight).stem.replace("-seg", "")
            online_dirs[variant] = output_dir.parent / f"{prefix}_swap_{stem}"
        else:
            online_dirs[variant] = output_dir.parent / f"{prefix}_{safe_name(variant)}"
    for directory in online_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    def finished(directory):
        return {path.stem.replace("upgraded_", "", 1) for path in directory.glob("upgraded_*.json")}

    done = finished(output_dir) if args.resume else set()
    for directory in online_dirs.values():
        done &= finished(directory) if args.resume else set()
    timing_path = output_dir.parent / f"{output_dir.name}_timing.jsonl"
    boxes_file = open(args.yolo_boxes, "a", encoding="utf-8") if args.yolo_boxes else None
    with timing_path.open("a" if args.resume else "w", encoding="utf-8") as timing_file:
        for index, image in enumerate(images, start=1):
            key = Path(image).stem
            if key in done:
                continue
            if args.stop_at and time.time() >= args.stop_at:
                print(f"[STOP] --stop-at reached before {key}; rerun with --resume to continue", flush=True)
                break
            pipeline.process_image(image, str(output_dir), **PIPELINE_SETTINGS)
            if boxes_file is not None:
                # Compact copy of the primary detector's raw output (what gt_eval needs), for runs
                # too large to keep the full YOLO cache with masks (about 1 MB per image).
                memo = pipeline.memo.get("yolo")
                detections = memo[1]["detections"] if memo is not None else []
                compact = [
                    {"box": [float(v) for v in item["box"]], "confidence": float(item["confidence"]),
                     "class_name": item["class_name"]}
                    for item in detections
                ]
                boxes_file.write(json.dumps({"key": key, "detections": compact}) + "\n")
                boxes_file.flush()
            if load_models:
                record = {"key": key, "index": index, **pipeline.timing}
                timing_file.write(json.dumps(record) + "\n")
                timing_file.flush()
            for variant, directory in online_dirs.items():
                if variant in swaps:
                    primary, primary_dir, save_yolo = pipeline.detectors, pipeline.cache.yolo_dir, pipeline.save_yolo
                    pipeline.detectors = swaps[variant]["detectors"]
                    pipeline.cache.yolo_dir = swaps[variant]["yolo_dir"]
                    pipeline.save_yolo = False  # swap detections stay in memory (disk budget)
                    try:
                        pipeline.process_image(image, str(directory), **PIPELINE_SETTINGS)
                    finally:
                        pipeline.detectors, pipeline.cache.yolo_dir, pipeline.save_yolo = primary, primary_dir, save_yolo
                else:
                    with variants.patched(variant, pipeline):
                        pipeline.process_image(image, str(directory), **PIPELINE_SETTINGS)
            pipeline.memo.clear()
    print(f"[DONE] {args.mode} variant={args.variant or 'full'} online={online} images={len(images)} -> {output_dir}")


def safe_name(name):
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def strip_report(report):
    report = dict(report)
    report.pop("visualization", None)
    report.pop("image", None)
    return report


def check(args):
    first = {p.name: p for p in Path(args.a).glob("upgraded_*.json")}
    second = {p.name: p for p in Path(args.b).glob("upgraded_*.json")}
    common = sorted(set(first) & set(second))
    mismatched = []
    for name in common:
        left = strip_report(json.loads(first[name].read_text(encoding="utf-8")))
        right = strip_report(json.loads(second[name].read_text(encoding="utf-8")))
        if left != right:
            mismatched.append(name)
    print(
        f"[CHECK] compared={len(common)} only_a={len(set(first) - set(second))} "
        f"only_b={len(set(second) - set(first))} mismatched={len(mismatched)}"
    )
    for name in mismatched[:10]:
        print("  MISMATCH", name)
    return 1 if mismatched or not common else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("infer", "replay"):
        p = sub.add_parser(mode)
        p.add_argument("--image-list", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--cache", default=str(DEFAULT_CACHE))
        p.add_argument("--yolo-weight", default=str(ROOT / CONFIG["yolo_weight"]))
        p.add_argument("--variant", default="full")
        p.add_argument("--limit", type=int)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--no-vis", action="store_true", help="Skip image/visualization files.")
        p.add_argument("--no-depth-cache", action="store_true",
                       help="Do not store raw depth (3.7 MB/image); such images cannot be replayed.")
        p.add_argument("--no-yolo-cache", action="store_true", help="Keep YOLO outputs in memory only.")
        p.add_argument("--variants", default="",
                       help="';'-separated variants run online on the same model outputs after the full system.")
        p.add_argument("--perturb", default="", help="Apply a perturb.py perturbation in memory when reading images.")
        p.add_argument("--stop-at", type=float, default=0.0,
                       help="Unix time after which no new image is started (clean stop for time-limited sessions).")
        p.add_argument("--yolo-boxes", default="",
                       help="Append the primary detector's boxes, scores and classes per image to this JSONL file.")
    p = sub.add_parser("check")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    args = parser.parse_args(argv)
    if args.mode == "check":
        return check(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
