"""Ablation variants for camera-ready replays.

Each variant patches one part of the fusion stage of ``test3`` inside the replay
process only; ``test3.py`` itself is never edited. The parameterised
re-implementations below copy the original code. Replaying them with the
published constants (``legacy_default``, ``risk_default``, ``sq_default``) must
reproduce the ``full`` reports exactly, which validates every variant built on
them.

Variant names
-------------
full                      original pipeline (no patch)
legacy_default            re-implemented object-depth reliability, default constants
risk_default              re-implemented risk/advisory mapping, gates on
sq_default                re-implemented scene-quality score, default weights
no_gates                  advisory = score cut-offs only (no severity gates)
single_region             object depth from the centre region only
no_scene_quality          fusion sees a neutral scene-quality score (1.0, GOOD)
no_hybridnets             dataset-average road prior instead of HybridNets
no_depth                  dataset-average depth prior instead of Depth Anything V2
B0..B3                    cumulative build-up baselines (fallback removed offline)
sens:key=value[,key=value] reliability constants, e.g. sens:stable_c=0.60
risk:scale=value          multiply the advisory score cut-offs (24/40/60/78)
sq:w1,w2,w3               scene-quality weights (illumination, contrast, sharpness)
A+B                       combine variants, e.g. no_gates+single_region
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import test3

PRIORS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "camera_ready" / "priors"
REGION_BOXES = {
    "center": (0.18, 0.82, 0.18, 0.82),
    "lower": (0.12, 0.88, 0.48, 0.88),
    "inner": (0.28, 0.72, 0.25, 0.72),
}
DEFAULT_LEGACY = {
    "regions": ("center", "lower", "inner"),
    "k_ratio": 1.35,
    "k_spread": 0.85,
    "n_support": 120.0,
    "stable_c": 0.68,
    "stable_a": 1.35,
    "check_c": 0.35,
    "check_a": 1.80,
}
DEFAULT_CUTOFFS = (78.0, 60.0, 40.0, 24.0)
DEFAULT_SQ_WEIGHTS = (0.40, 0.30, 0.30)
ORIGINAL_SCENE_QUALITY = test3.assess_scene_quality
ACTION_SEVERITY = test3.ACTION_SEVERITY
VULNERABLE_CLASSES = test3.VULNERABLE_CLASSES


# ---------------------------------------------------------------------------
# Object-depth reliability (copy of ADASPipeline.get_object_legacy_unit)
# ---------------------------------------------------------------------------
def make_legacy_unit(
    regions,
    k_ratio,
    k_spread,
    n_support,
    stable_c,
    stable_a,
    check_c,
    check_a,
):
    region_specs = [(*REGION_BOXES[name], name) for name in regions]

    def get_object_legacy_unit(legacy_distance_map, object_mask, box, mask_source, mask_quality):
        x1, y1, x2, y2 = box
        box_width = max(x2 - x1, 1)
        box_height = max(y2 - y1, 1)
        erode_size = max(3, int(min(box_width, box_height) * 0.045))
        if erode_size % 2 == 0:
            erode_size += 1
        core_mask = cv2.erode(
            object_mask.astype(np.uint8),
            np.ones((erode_size, erode_size), np.uint8),
            iterations=1,
        )
        if core_mask.sum() < 20:
            core_mask = object_mask.astype(np.uint8)

        region_estimates = []
        all_values = []
        for x_start, x_end, y_start, y_end, name in region_specs:
            region_mask = np.zeros_like(core_mask)
            rx1 = x1 + int(box_width * x_start)
            rx2 = x1 + int(box_width * x_end)
            ry1 = y1 + int(box_height * y_start)
            ry2 = y1 + int(box_height * y_end)
            region_mask[ry1:ry2, rx1:rx2] = core_mask[ry1:ry2, rx1:rx2]
            values = legacy_distance_map[region_mask == 1]
            values = values[np.isfinite(values) & (values > 0.1) & (values < 999.0)]
            if values.size < 10:
                continue
            low, high = np.percentile(values, [12, 88])
            values = values[(values >= low) & (values <= high)]
            if values.size == 0:
                continue
            estimate = float(np.median(values))
            region_estimates.append((name, estimate, int(values.size)))
            all_values.append(values)

        if not region_estimates:
            values = legacy_distance_map[object_mask]
            values = values[np.isfinite(values) & (values > 0.1) & (values < 999.0)]
            if values.size == 0:
                return {
                    "value": None,
                    "min": None,
                    "max": None,
                    "confidence": 0.0,
                    "status": "UNAVAILABLE",
                    "regions": [],
                }
            region_estimates = [("fallback", float(np.median(values)), int(values.size))]
            all_values = [values]

        estimates = np.asarray([item[1] for item in region_estimates], dtype=np.float64)
        weights = np.asarray([min(item[2], 300) for item in region_estimates], dtype=np.float64)
        center = float(np.average(np.log(estimates), weights=weights))
        value = float(np.exp(center))
        agreement_ratio = float(estimates.max() / max(estimates.min(), 1e-6))
        values = np.concatenate(all_values)
        q20, q80 = np.percentile(values, [20, 80])
        relative_spread = float((q80 - q20) / max(value, 1e-6))

        confidence = float(
            np.clip(k_ratio / agreement_ratio, 0.0, 1.0)
            * np.exp(-relative_spread * k_spread)
            * np.clip(values.size / n_support, 0.25, 1.0)
        )
        if mask_source != "segmentation":
            confidence *= 0.58
        confidence *= float(np.clip(0.45 + mask_quality * 0.75, 0.35, 1.0))
        if x1 <= 1 or y1 <= 1 or x2 >= legacy_distance_map.shape[1] - 1:
            confidence *= 0.72
        if y2 >= legacy_distance_map.shape[0] - 1:
            confidence *= 0.82

        if confidence >= stable_c and agreement_ratio <= stable_a:
            status = "STABLE"
        elif confidence >= check_c and agreement_ratio <= check_a:
            status = "CHECK"
        else:
            status = "UNCERTAIN"

        uncertainty = float(
            np.clip(
                0.10 + relative_spread * 0.48 + (1.0 - confidence) * 0.35,
                0.16,
                0.80,
            )
        )
        return {
            "value": value,
            "min": max(0.1, min(value * (1.0 - uncertainty), float(q20))),
            "max": min(999.0, max(value * (1.0 + uncertainty), float(q80))),
            "confidence": confidence,
            "status": status,
            "agreement_ratio": agreement_ratio,
            "regions": [
                {"name": name, "unit": round(estimate, 2), "pixels": pixels}
                for name, estimate, pixels in region_estimates
            ],
        }

    return get_object_legacy_unit


def patch_legacy(**overrides):
    params = dict(DEFAULT_LEGACY)
    params.update(overrides)
    test3.ADASPipeline.get_object_legacy_unit = staticmethod(make_legacy_unit(**params))


# ---------------------------------------------------------------------------
# Risk score and advisory (copy of ADASPipeline.assess_risk)
# ---------------------------------------------------------------------------
def make_assess_risk(gates=True, cutoffs=DEFAULT_CUTOFFS):
    stop_cut, brake_cut, slow_cut, monitor_cut = cutoffs

    def map_score(score):
        if score >= stop_cut:
            return "STOP"
        if score >= brake_cut:
            return "BRAKE"
        if score >= slow_cut:
            return "SLOW"
        if score >= monitor_cut:
            return "MONITOR"
        return "CLEAR"

    def assess_risk(
        distance_m,
        relative_distance,
        legacy_distance,
        legacy_confidence,
        legacy_status,
        distance_confidence,
        detection_confidence,
        scene_quality_score,
        relation,
        class_name,
        box,
        image_shape,
    ):
        height, width = image_shape[:2]
        x1, y1, x2, y2 = box
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(height * width, 1)
        bottom_ratio = y2 / max(height, 1)

        relative_score = (
            (100.0 - relative_distance) * 0.45
            if relative_distance is not None
            else 0.0
        )
        if legacy_distance is None:
            legacy_score = 0.0
        elif legacy_distance <= 4.0:
            legacy_score = 54.0
        elif legacy_distance <= 12.0:
            legacy_score = float(np.interp(legacy_distance, [4.0, 12.0], [54.0, 36.0]))
        elif legacy_distance <= 30.0:
            legacy_score = float(np.interp(legacy_distance, [12.0, 30.0], [36.0, 16.0]))
        elif legacy_distance <= 80.0:
            legacy_score = float(np.interp(legacy_distance, [30.0, 80.0], [16.0, 3.0]))
        else:
            legacy_score = 0.0
        legacy_score *= 0.35 + 0.65 * legacy_confidence

        if distance_m is not None:
            if distance_m <= 3.0:
                metric_score = 56.0
            elif distance_m <= 7.0:
                metric_score = float(np.interp(distance_m, [3.0, 7.0], [56.0, 46.0]))
            elif distance_m <= 15.0:
                metric_score = float(np.interp(distance_m, [7.0, 15.0], [46.0, 30.0]))
            elif distance_m <= 30.0:
                metric_score = float(np.interp(distance_m, [15.0, 30.0], [30.0, 14.0]))
            elif distance_m <= 60.0:
                metric_score = float(np.interp(distance_m, [30.0, 60.0], [14.0, 4.0]))
            else:
                metric_score = max(0.0, 4.0 - (distance_m - 60.0) * 0.04)
            score = metric_score * (0.70 + 0.30 * distance_confidence)
            score = max(score, relative_score * 0.45, legacy_score * 0.70)
        else:
            score = max(relative_score, legacy_score)

        score += min(14.0, area_ratio * 170.0)
        score += max(0.0, bottom_ratio - 0.45) * 14.0
        if relation == "PATH":
            score += 32.0
        elif relation == "EDGE":
            score += 15.0
        if class_name in VULNERABLE_CLASSES:
            score += 9.0
        if relation in {"PATH", "EDGE"} and distance_confidence < 0.35:
            score += 4.0

        distance_evidence = max(float(legacy_confidence), float(distance_confidence))
        evidence_confidence = float(
            np.sqrt(max(float(detection_confidence), 0.0) * distance_evidence)
            * (0.65 + 0.35 * float(scene_quality_score))
        )

        if not gates:
            return float(np.clip(score, 0.0, 100.0)), map_score(score), evidence_confidence

        if relation == "SIDE":
            if (
                class_name in VULNERABLE_CLASSES
                and (
                    (distance_m is not None and distance_m <= 12.0)
                    or (relative_distance is not None and relative_distance <= 45)
                    or (
                        legacy_distance is not None
                        and legacy_confidence >= 0.35
                        and legacy_distance <= 12.0
                    )
                )
                and bottom_ratio >= 0.70
            ):
                action = "SLOW"
                if (
                    detection_confidence < 0.25
                    and legacy_status in {"UNCERTAIN", "UNAVAILABLE"}
                ):
                    action = "MONITOR"
                return min(float(score), 45.0), action, evidence_confidence
            side_score = min(float(score), 39.0)
            return (
                side_score,
                "MONITOR" if side_score >= monitor_cut else "CLEAR",
                evidence_confidence,
            )

        if relation == "EDGE":
            score = min(score, 69.0)

        action = map_score(score)

        immediate_metric = (
            distance_m is not None
            and distance_confidence >= 0.55
            and distance_m <= 4.0
        )
        immediate_legacy = (
            legacy_distance is not None
            and legacy_confidence >= 0.60
            and legacy_distance <= 4.0
            and area_ratio >= 0.025
            and bottom_ratio >= 0.68
        )
        if action == "STOP" and not (
            relation == "PATH" and (immediate_metric or immediate_legacy)
        ):
            action = "BRAKE"

        credible_brake = (
            relation in {"PATH", "EDGE"}
            and (
                area_ratio >= 0.005
                or bottom_ratio >= 0.68
                or (
                    legacy_distance is not None
                    and legacy_confidence >= 0.45
                    and legacy_distance <= 14.0
                )
            )
        )
        if action == "BRAKE" and not credible_brake:
            action = "SLOW"

        if relation == "EDGE" and action == "STOP":
            action = "BRAKE"

        if action == "STOP" and (
            detection_confidence < 0.55
            or evidence_confidence < 0.48
            or legacy_status != "STABLE"
        ):
            action = "BRAKE"
        if action == "BRAKE" and (
            detection_confidence < 0.30 or evidence_confidence < 0.28
        ):
            action = "SLOW"
        if (
            legacy_status in {"UNCERTAIN", "UNAVAILABLE"}
            and detection_confidence < 0.35
            and ACTION_SEVERITY[action] > ACTION_SEVERITY["MONITOR"]
        ):
            action = "MONITOR"
        if scene_quality_score < 0.35:
            if action == "BRAKE" and (
                detection_confidence < 0.45 or legacy_status != "STABLE"
            ):
                action = "SLOW"
            if (
                action == "SLOW"
                and detection_confidence < 0.25
                and legacy_status != "STABLE"
            ):
                action = "MONITOR"
            if (
                action == "SLOW"
                and detection_confidence < 0.30
                and legacy_distance is not None
                and legacy_distance > 30.0
            ):
                action = "MONITOR"
        return float(np.clip(score, 0.0, 100.0)), action, evidence_confidence

    return assess_risk


def patch_risk(gates=True, cutoffs=DEFAULT_CUTOFFS):
    test3.ADASPipeline.assess_risk = staticmethod(make_assess_risk(gates, tuple(cutoffs)))


# ---------------------------------------------------------------------------
# Scene quality (copy of test3.assess_scene_quality)
# ---------------------------------------------------------------------------
def make_scene_quality(weights=DEFAULT_SQ_WEIGHTS):
    w_illum, w_contrast, w_sharp = weights

    def assess_scene_quality(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        p10, median, p90 = np.percentile(gray, [10, 50, 90])
        dynamic_range = float(p90 - p10)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare_fraction = float(np.mean(gray >= 245))

        illumination_score = float(np.clip((median - 18.0) / 72.0, 0.0, 1.0))
        contrast_score = float(np.clip((dynamic_range - 20.0) / 80.0, 0.0, 1.0))
        sharpness_score = float(
            np.clip(
                (np.log1p(laplacian_variance) - np.log1p(12.0))
                / (np.log1p(500.0) - np.log1p(12.0)),
                0.0,
                1.0,
            )
        )
        score = float(
            w_illum * illumination_score
            + w_contrast * contrast_score
            + w_sharp * sharpness_score
        )

        flags = []
        if median < 45.0:
            flags.append("LOW_LIGHT")
        if dynamic_range < 55.0:
            flags.append("LOW_CONTRAST")
        if laplacian_variance < 60.0:
            flags.append("BLUR")
        if glare_fraction > 0.025 and median < 90.0:
            flags.append("GLARE")

        if score < 0.35 or ("LOW_LIGHT" in flags and "BLUR" in flags):
            status = "DEGRADED"
        elif score < 0.58 or flags:
            status = "LIMITED"
        else:
            status = "GOOD"
        return {
            "status": status,
            "score": score,
            "flags": flags,
            "median_brightness": float(median),
            "dynamic_range_p10_p90": dynamic_range,
            "laplacian_variance": laplacian_variance,
            "glare_fraction": glare_fraction,
        }

    return assess_scene_quality


def patch_scene_quality(weights=DEFAULT_SQ_WEIGHTS):
    test3.assess_scene_quality = make_scene_quality(tuple(weights))


def patch_neutral_scene_quality():
    def neutral(image):
        quality = dict(ORIGINAL_SCENE_QUALITY(image))
        quality.update({"status": "GOOD", "score": 1.0, "flags": []})
        return quality

    test3.assess_scene_quality = neutral


# ---------------------------------------------------------------------------
# Component priors (computed on the dev split by priors.py)
# ---------------------------------------------------------------------------
def _resize_to(array, shape, interpolation):
    if array.shape[:2] == tuple(shape[:2]):
        return array.copy()
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)


ORIGINAL_CALIBRATION = test3.ADASPipeline.calibrate_metric_depth


def patch_safe_calibration():
    """Prior inputs can make the flat-road least-squares fit degenerate.

    The original function already returns ``(None, info)`` when calibration is
    impossible; this wrapper routes a numerical failure to that same branch. It
    is applied only in the prior-based ablations, never to the full system.
    """

    def calibrate_metric_depth(raw_depth, road_mask, ego_corridor, camera_height_m, focal_px, horizon_y):
        try:
            return ORIGINAL_CALIBRATION(raw_depth, road_mask, ego_corridor, camera_height_m, focal_px, horizon_y)
        except np.linalg.LinAlgError:
            return None, {
                "method": "flat-road-depth-anything",
                "status": "unavailable",
                "confidence": 0.0,
                "sample_rows": 0,
                "horizon_y": round(float(horizon_y), 2),
                "focal_px": round(float(focal_px), 2),
                "camera_height_m": round(float(camera_height_m), 3),
            }

    test3.ADASPipeline.calibrate_metric_depth = staticmethod(calibrate_metric_depth)


def patch_road_prior(pipeline):
    patch_safe_calibration()
    road_prior = np.load(PRIORS_DIR / "road_prior.npy")

    def road_override(image):
        road = _resize_to(road_prior, image.shape, cv2.INTER_NEAREST).astype(np.uint8)
        return road, np.zeros_like(road)

    pipeline.road_override = road_override


def patch_depth_prior(pipeline):
    patch_safe_calibration()
    depth_prior = np.load(PRIORS_DIR / "depth_prior.npy").astype(np.float32)

    def depth_override(image):
        return _resize_to(depth_prior, image.shape, cv2.INTER_LINEAR).astype(np.float32)

    pipeline.depth_override = depth_override


# ---------------------------------------------------------------------------
# Exploratory (not part of the published method): restrict the perspective
# ego corridor to the dilated HybridNets drivable mask. Used only to discuss
# the distant-object PATH failure mode found by the ground-truth evaluation.
# ---------------------------------------------------------------------------
ORIGINAL_EGO_CORRIDOR = test3.ADASPipeline.get_ego_corridor


def patch_corridor_road(fraction=0.05):
    def get_ego_corridor(road_mask, lane_mask):
        corridor = ORIGINAL_EGO_CORRIDOR(road_mask, lane_mask)
        size = max(3, int(road_mask.shape[0] * fraction)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        drivable = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=1)
        return (corridor.astype(np.uint8) & drivable).astype(np.uint8)

    test3.ADASPipeline.get_ego_corridor = staticmethod(get_ego_corridor)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _parse_value(text):
    try:
        return float(text)
    except ValueError:
        return tuple(text.split("|"))


def apply_one(name, pipeline):
    if name == "full":
        return
    if name == "legacy_default":
        patch_legacy()
    elif name == "risk_default":
        patch_risk(gates=True)
    elif name == "sq_default":
        patch_scene_quality()
    elif name == "no_gates":
        patch_risk(gates=False)
    elif name == "single_region":
        patch_legacy(regions=("center",))
    elif name == "no_scene_quality":
        patch_neutral_scene_quality()
    elif name == "no_hybridnets":
        patch_road_prior(pipeline)
    elif name == "no_depth":
        patch_depth_prior(pipeline)
    elif name == "corridor_road":
        patch_corridor_road()
    elif name.startswith("sens:"):
        overrides = {}
        for item in name[5:].split(","):
            key, value = item.split("=")
            overrides[key] = _parse_value(value)
        patch_legacy(**overrides)
    elif name.startswith("risk:"):
        key, value = name[5:].split("=")
        if key != "scale":
            raise ValueError(name)
        patch_risk(gates=True, cutoffs=[c * float(value) for c in DEFAULT_CUTOFFS])
    elif name.startswith("sq:"):
        patch_scene_quality([float(v) for v in name[3:].split(",")])
    else:
        raise ValueError(f"Unknown variant: {name}")


BUILD_UP = {
    "B0": ["no_hybridnets", "no_depth", "no_scene_quality", "no_gates"],
    "B1": ["no_depth", "no_scene_quality", "no_gates"],
    "B2": ["single_region", "no_scene_quality", "no_gates"],
    "B3": ["no_gates"],
}


def apply(name, pipeline):
    parts = BUILD_UP.get(name, None) or name.split("+")
    for part in parts:
        apply_one(part, pipeline)


PATCHED_METHODS = ("get_object_legacy_unit", "assess_risk", "calibrate_metric_depth", "get_ego_corridor")


class patched:
    """Apply a variant for the duration of a ``with`` block, then restore.

    Lets one inference process run the full system and several ablations on
    the same in-memory model outputs, without caching the depth maps.
    """

    def __init__(self, name, pipeline):
        self.name = name
        self.pipeline = pipeline

    def __enter__(self):
        self.methods = {key: test3.ADASPipeline.__dict__[key] for key in PATCHED_METHODS}
        self.scene_quality = test3.assess_scene_quality
        self.overrides = (self.pipeline.road_override, self.pipeline.depth_override)
        if self.name != "full":
            apply(self.name, self.pipeline)
        return self

    def __exit__(self, *exc):
        for key, value in self.methods.items():
            setattr(test3.ADASPipeline, key, value)
        test3.assess_scene_quality = self.scene_quality
        self.pipeline.road_override, self.pipeline.depth_override = self.overrides
        return False
