"""Paper figures (static, print/grayscale-safe).

fig_sensitivity : small multiples, one panel per metric (one x-axis each), rows
                  are the fusion constants changed by -20 % / +20 % on the dev
                  split; the vertical rule is the published-constant value.
fig_qualitative : 2x3 grid of hard-set cases chosen by fixed rules (three
                  intended behaviours, three failure modes) with GT and
                  predicted boxes.
fig_regions     : one object crop with the three depth-sampling regions.

Palette: categorical slots 1-3 of the reference palette (blue, orange, aqua),
validated all-pairs with scripts/validate_palette.js; identity is always also
carried by marker shape or a text label.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT = ROOT / "outputs" / "camera_ready"
FIG = OUT / "figures"
RUNS = OUT / "runs"
EVAL = OUT / "eval"
VAL = ROOT.parent / "data" / "images" / "100k" / "val"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RELATION_COLOR = {"PATH": ORANGE, "EDGE": AQUA, "SIDE": BLUE}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SENS_ROWS = (
    ("Stable conf. $\\tau_s$ (0.68)", "sens:stable_c=0.544", "sens:stable_c=0.816"),
    ("Stable ratio $\\alpha_s$ (1.35)", "sens:stable_a=1.08", "sens:stable_a=1.62"),
    ("Check conf. $\\tau_c$ (0.35)", "sens:check_c=0.28", "sens:check_c=0.42"),
    ("Check ratio $\\alpha_c$ (1.80)", "sens:check_a=1.44", "sens:check_a=2.16"),
    ("Agreement $k_a$ (1.35)", "sens:k_ratio=1.08", "sens:k_ratio=1.62"),
    ("Spread $k_s$ (0.85)", "sens:k_spread=0.68", "sens:k_spread=1.02"),
    ("Support $N_0$ (120)", "sens:n_support=96", "sens:n_support=144"),
    ("Risk cut-offs", "risk:scale=0.8", "risk:scale=1.2"),
)
SENS_METRICS = (
    ("Stable\nratio (%)", lambda e: 100 * e["reliability"]["stable_ratio_wilson"]["value"]),
    ("BRAKE/STOP\nimages (%)", lambda e: 100 * e["advisory"]["severe"]["value"]),
    ("Over-\ncaution (%)", lambda e: 100 * e["advisory"]["over_caution"]["value"]),
)


def style_axis(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_sensitivity(results):
    sens = results.get("sensitivity", {})
    if "default" not in sens:
        print("[FIG] sensitivity skipped (no dev results)")
        return
    rows = [row for row in SENS_ROWS if row[1] in sens and row[2] in sens]
    fig, axes = plt.subplots(1, len(SENS_METRICS), figsize=(3.45, 0.22 * len(rows) + 0.85), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (title, getter) in zip(axes, SENS_METRICS):
        base = getter(sens["default"])
        low = [getter(sens[r[1]]) for r in rows]
        high = [getter(sens[r[2]]) for r in rows]
        ax.axvline(base, color=INK2, linewidth=0.8, zorder=1)
        for yi, lo, hi in zip(y, low, high):
            ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=1.2, zorder=1, solid_capstyle="round")
        ax.scatter(low, y, s=16, marker="o", facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=3, label="$-20\\%$")
        ax.scatter(high, y, s=16, marker="^", color=ORANGE, edgecolor="white", linewidth=0.4, zorder=3, label="$+20\\%$")
        ax.set_title(title, fontsize=6.5, color=INK, pad=3, linespacing=1.0)
        style_axis(ax)
        span = max(max(low + high + [base]) - min(low + high + [base]), 1.0)
        ax.set_xlim(min(low + high + [base]) - 0.15 * span, max(low + high + [base]) + 0.15 * span)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.62, -0.01), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
    save(fig, "fig_sensitivity")


def fig_operating(results):
    curve = results.get("operating") or {}
    if not curve:
        print("[FIG] operating skipped")
        return
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    styles = {"dev": (BLUE, "o", "Dev (500, selection)"), "standard": (ORANGE, "^", "1k test subset"),
              "valop": (ORANGE, "^", "Held-out val_rest (8.4k)"), "train": (AQUA, "s", "Train split (70k)")}
    # Label offsets per split keep the scale annotations apart (dev below-left, held-out above-right).
    # The train curve lies on the held-out one at the same scales, so it carries no labels (they
    # would cover the dev points).
    offsets = {"dev": (5, -9), "standard": (4, 4), "valop": (4, 4)}
    # The 8.4k held-out curve replaces the 1k subset curve when it exists (same role, 8x the images).
    order = [s for s in ("dev", "valop", "train") if s in curve] if "valop" in curve else list(curve)
    for split in order:
        points = curve[split]
        color, marker, label = styles.get(split, (AQUA, "s", split))
        points = sorted(points, key=lambda p: p["scale"])
        x = [100 * p["over_caution"] / max(p["free"], 1) for p in points]
        y = [100 * p["unsafe_clear"] / max(p["occupied"], 1) for p in points]
        ax.plot(x, y, color=color, linewidth=1.0, zorder=2)
        ax.scatter(x, y, s=14, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=3, label=label)
        offset = offsets.get(split)
        for p, xi, yi in zip(points, x, y):
            if p["scale"] == 1.0:
                ax.scatter([xi], [yi], s=70, marker="o", facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4)
            if offset and p["scale"] in (1.4, 2.0, 2.5):
                ax.annotate(f"$\\times${p['scale']:g}", (xi, yi), textcoords="offset points", xytext=offset,
                            fontsize=6, color=color, ha="right" if offset[0] < 0 else "left")
    ax.scatter([], [], s=70, marker="o", facecolor="none", edgecolor=INK, linewidth=0.9, label="published ($\\times$1)")
    ax.set_xlabel("Over-caution: SLOW or higher on a free ego lane (%)")
    ax.set_ylabel("Unsafe CLEAR\n(% of occupied)")
    style_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_operating")


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_report(key):
    return json.loads((RUNS / "hard_full" / f"upgraded_{key}.json").read_text(encoding="utf-8"))


SEVERITY = {"CLEAR": 0, "MONITOR": 1, "SLOW": 2, "BRAKE": 3, "STOP": 4}


def pick_cases():
    images = {row["key"]: row for row in read_csv(EVAL / "hard_full" / "images.csv")}
    objects = read_csv(EVAL / "hard_full" / "objects.csv")
    no_gates_path = EVAL / "hard_no_gates" / "images.csv"
    no_gates = {row["key"]: row for row in read_csv(no_gates_path)} if no_gates_path.exists() else {}
    by_key = {}
    for row in objects:
        by_key.setdefault(row["key"], []).append(row)
    keys = sorted(images)

    def first(predicate, score=lambda k: 0):
        candidates = [k for k in keys if predicate(k)]
        return max(candidates, key=score) if candidates else None

    cases = []
    used = set()

    def add(title, key):
        if key and key not in used:
            used.add(key)
            cases.append((title, key))

    add(
        "(a) Dense scene: correct PATH objects",
        first(
            lambda k: images[k]["category"] == "complex" and images[k]["severe_unsupported"] == "0",
            lambda k: sum(1 for o in by_key.get(k, []) if o["tp"] == "1" and o["state"] == "STABLE"),
        ),
    )
    def capped(k):
        other = no_gates.get(k)
        return bool(other) and SEVERITY[other["advisory"]] > SEVERITY[images[k]["advisory"]]

    key_b = first(
        lambda k: images[k]["scene_status"] == "DEGRADED" and capped(k) and int(images[k]["n_tp"]) > 0,
        lambda k: int(images[k]["n_tp"]),
    )
    if key_b:
        add(f"(b) Degraded night: gates cap {no_gates[key_b]['advisory']}", key_b)
    add(
        "(c) Detector-empty: fallback MONITOR",
        first(
            lambda k: images[k]["n_pred"] == "0" and images[k]["fallback_fired"] == "1",
            lambda k: int(images[k]["n_gt"]),
        ),
    )
    add(
        "(d) Failure: distant parked car as PATH",
        first(
            lambda k: images[k]["severe_unsupported"] == "1"
            and any(o["relation"] == "PATH" and o["tp"] == "1" and o["gt_lane"] == "off" for o in by_key.get(k, [])),
            lambda k: sum(1 for o in by_key.get(k, []) if o["relation"] == "PATH" and o["gt_lane"] == "off"),
        ),
    )
    add(
        "(e) Disagreement: annotated lane spans the street",
        first(
            lambda k: any(o["relation"] == "SIDE" and o["gt_lane"] == "direct" and o["gt_relevant"] == "1" for o in by_key.get(k, [])),
            lambda k: sum(1 for o in by_key.get(k, []) if o["relation"] == "SIDE" and o["gt_lane"] == "direct"),
        ),
    )
    add(
        "(f) Truncated object kept at CHECK",
        first(
            lambda k: any(o["state"] == "CHECK" and o["gt_truncated"] == "1" for o in by_key.get(k, [])),
            lambda k: sum(1 for o in by_key.get(k, []) if o["state"] == "CHECK"),
        ),
    )
    return cases


def draw_case(ax, title, key, gt_records):
    from gt_eval import load_drivable

    image = cv2.cvtColor(cv2.imread(str(VAL / f"{key}.jpg")), cv2.COLOR_BGR2RGB)
    report = load_report(key)
    record = gt_records[key]
    drivable = load_drivable(record)
    overlay = image.copy()
    overlay[drivable == 0] = (0.6 * overlay[drivable == 0] + 0.4 * np.array([120, 200, 120])).astype(np.uint8)
    ax.imshow(overlay)
    for obj in record["objects"]:
        x1, y1, x2, y2 = obj["box"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="black", linewidth=1.6))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=0.7))
    height, width = image.shape[:2]
    min_label_height = 0.12 * height
    char_w, tag_h = 0.026 * width, 0.08 * height  # tag size in image pixels at the printed font size
    placed = []
    # Largest boxes first; a tag that would cover another tag is left out (the box keeps its color).
    for det in sorted(report["detections"], key=lambda d: -(d["box_xyxy"][3] - d["box_xyxy"][1])):
        x1, y1, x2, y2 = det["box_xyxy"]
        color = RELATION_COLOR[det["corridor_relation"]]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.4))
        if y2 - y1 < min_label_height:
            continue
        state = det["original_unit_status"]
        tag = det["corridor_relation"] + ("" if state == "STABLE" else ", " + state)
        tx = min(max(x1, 0), width - len(tag) * char_w)
        rect = (tx, y1 - tag_h, tx + len(tag) * char_w, y1)
        if any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3] for r in placed):
            continue
        placed.append(rect)
        ax.text(tx + 2, y1 - 4, tag, color="white", fontsize=5.0, fontweight="bold",
                bbox={"boxstyle": "square,pad=0.1", "facecolor": color, "edgecolor": "none"})
    advisory = report["global_advisory"]
    scene = report["scene_quality"]["status"]
    ax.set_title(f"{title}\n{advisory} | scene {scene} | {len(report['detections'])} obj.", fontsize=6.5, color=INK, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_qualitative():
    from gt_eval import load_gt

    cases = pick_cases()
    if not cases:
        print("[FIG] qualitative skipped")
        return
    gt_records = load_gt([key for _, key in cases])
    from gt_eval import ground_lane, load_drivable

    for key in gt_records:
        drivable = load_drivable(gt_records[key])
        for obj in gt_records[key]["objects"]:
            obj["lane"] = ground_lane(drivable, obj["box"])
    rows = 2 if len(cases) > 3 else 1
    fig, axes = plt.subplots(rows, 3, figsize=(7.16, 1.78 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, key) in zip(axes, cases):
        draw_case(ax, title, key, gt_records)
    for ax in axes[len(cases):]:
        ax.axis("off")
    fig.legend(handles=qualitative_legend(), loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.2, w_pad=0.3)
    save(fig, "fig_qualitative")
    (FIG / "fig_qualitative_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    fig_overlay_cases(cases)


# Six-page paper: three hard-set cases drawn with the system's own overlay (box color = the
# advisory of that object, short labels), large enough to read in print. Letters of the six-panel
# figure, title used in the paper.
OVERLAY_CASES = (("a", "Dense street: taxi ahead in the path"), ("b", "Degraded night: gates cap STOP"),
                 ("d", "Failure: parked car labeled PATH"))


def fig_overlay_cases(cases):
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from paper_visualization import render_overlay

    by_letter = {title[1]: key for title, key in cases}
    chosen = [(title, by_letter[letter]) for letter, title in OVERLAY_CASES if letter in by_letter]
    if len(chosen) != len(OVERLAY_CASES):
        print("[FIG] overlay cases skipped")
        return
    fig, axes = plt.subplots(1, len(chosen), figsize=(7.16, 1.52))
    for index, (ax, (title, key)) in enumerate(zip(axes, chosen)):
        report = load_report(key)
        image = cv2.imread(str(VAL / f"{key}.jpg"))
        masks = np.load(OUT / "cache" / "hybridnets" / f"{key}.npz")
        overlay = render_overlay(image, masks["road"], masks["lane"], report["detections"], report["global_advisory"],
                                 None, None, report["scene_quality"], None, text_scale=2.6, max_labels=3, compact=True)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"({'abc'[index]}) {title}", fontsize=7, color=INK, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.35)
    save(fig, "fig_qualitative_main", pdf_dpi=400)


def qualitative_legend():
    return [
        plt.Line2D([0], [0], color=ORANGE, linewidth=1.6, label="PATH"),
        plt.Line2D([0], [0], color=AQUA, linewidth=1.6, label="EDGE"),
        plt.Line2D([0], [0], color=BLUE, linewidth=1.6, label="SIDE"),
        plt.Line2D([0], [0], color="black", linewidth=1.6, label="BDD100K box (white/black)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#78c878", edgecolor="none", label="GT ego lane (direct)"),
    ]


REGIONS = {
    "center": ("center", 0.18, 0.82, 0.18, 0.82, BLUE, "-"),
    "lower": ("lower", 0.12, 0.88, 0.48, 0.88, ORANGE, "--"),
    "inner": ("inner", 0.28, 0.72, 0.25, 0.72, AQUA, ":"),
}


def cached_mask(key, box):
    import pickle

    path = next((OUT / "cache" / "yolo").glob("yolo11x-seg-*"), None)
    if path is None or not (path / f"{key}.pkl").exists():
        return None
    payload = pickle.loads((path / f"{key}.pkl").read_bytes())
    from gt_eval import box_iou

    best, best_iou = None, 0.5
    for det in payload["detections"]:
        value = box_iou([float(v) for v in det["box"]], [float(v) for v in box])
        if value > best_iou and det["mask"] is not None and det["mask"]["kind"] == "bits":
            best, best_iou = det, value
    if best is None:
        return None
    mask = best["mask"]
    size = int(np.prod(mask["shape"]))
    return np.unpackbits(mask["bits"])[:size].reshape(mask["shape"]).astype(bool)


def region_candidates(state, require_tp=True, min_area=0.008):
    objects = read_csv(EVAL / "hard_full" / "objects.csv")
    by_key = {}
    for row in objects:
        by_key.setdefault(row["key"], []).append(row)
    out = []
    for key, rows in by_key.items():
        report = load_report(key)
        for det, row in zip(report["detections"], rows):
            x1, y1, x2, y2 = det["box_xyxy"]
            regions = {r["name"]: r["unit"] for r in det["original_unit_regions"]}
            if det["original_unit_status"] != state or (require_tp and row["tp"] != "1") or len(regions) != 3:
                continue
            if x1 <= 2 or y1 <= 2 or x2 >= 1277 or y2 >= 717:
                continue  # keep border truncation out of the illustration
            area = (x2 - x1) * (y2 - y1) / (1280 * 720)
            ratio = max(regions.values()) / min(regions.values())
            if min_area <= area <= 0.08:
                out.append((key, det, ratio, area))
    return out


def draw_region_panel(ax, key, det, title):
    image = cv2.cvtColor(cv2.imread(str(VAL / f"{key}.jpg")), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = det["box_xyxy"]
    pad = int(0.12 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
    ax.imshow(image[cy1:cy2, cx1:cx2])
    mask = cached_mask(key, det["box_xyxy"])
    if mask is not None:
        crop_mask = mask[cy1:cy2, cx1:cx2].astype(np.uint8)
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = contour[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), color="white", linewidth=0.7)
    width, height = x2 - x1, y2 - y1
    values = {r["name"]: r["unit"] for r in det["original_unit_regions"]}
    for label, (name, xs, xe, ys, ye, color, style) in REGIONS.items():
        ax.add_patch(plt.Rectangle((x1 - cx1 + width * xs, y1 - cy1 + height * ys), width * (xe - xs),
                                   height * (ye - ys), fill=False, edgecolor=color, linewidth=1.2, linestyle=style))
    ratio = max(values.values()) / min(values.values())
    text = " / ".join(f"{values[REGIONS[l][0]]:.1f}" for l in REGIONS)
    ax.set_title(f"{title}: $a$={ratio:.2f}, $c_u$={det['original_unit_confidence']:.2f}\n$u$ = {text}",
                 fontsize=6.3, color=INK, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_regions():
    stable = sorted(
        [c for c in region_candidates("STABLE") if c[2] <= 1.12 and c[1]["class"] == "car"],
        key=lambda c: (-c[3], c[0]),
    )
    check_all = region_candidates("CHECK", require_tp=False, min_area=0.003)
    # Prefer a readable crop (>= 0.6 % of the frame) with the largest regional disagreement.
    check = sorted([c for c in check_all if c[3] >= 0.006] or check_all, key=lambda c: (-c[2], c[0]))
    if not stable or not check:
        print("[FIG] regions skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.5))
    draw_region_panel(axes[0], stable[0][0], stable[0][1], "Stable")
    draw_region_panel(axes[1], check[0][0], check[0][1], "Check")
    handles = [plt.Line2D([0], [0], color=c, linestyle=s, linewidth=1.2, label=label)
               for label, (_, _, _, _, _, c, s) in REGIONS.items()]
    handles.append(plt.Line2D([0], [0], color="#9a9893", linewidth=0.8, label="instance mask"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.5)
    save(fig, "fig_regions")
    cases = {"stable": [stable[0][0], stable[0][1]["object_id"]], "check": [check[0][0], check[0][1]["object_id"]]}
    (FIG / "fig_regions_case.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")


def fig_pipeline():
    """Data flow of the method (vector, column width). Scene quality is computed from the image
    itself; the path relation needs the detector and HybridNets; the depth reliability needs the
    detector masks and the depth field; the fallback reads the depth along the corridor."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(3.45, 1.53))
    ax.set_xlim(0, 3.45)
    ax.set_ylim(0, 1.53)
    ax.axis("off")
    fills = {"input": "#e8eef7", "model": "#e2f0e4", "reason": "#fbf3d6", "fusion": "#fde4d0", "output": "#f8d7d6"}

    def box(x, y, w, h, title, sub=None, kind="model", dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045", linewidth=0.6,
                                    edgecolor=INK2, facecolor="white" if dashed else fills[kind],
                                    linestyle=(0, (3, 2)) if dashed else "-"))
        if sub is None:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=5.6, color=INK, linespacing=1.15)
        else:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=5.8, color=INK, fontweight="bold")
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=5.0, color=INK2)

    def route(points, dashed=False):
        style = (0, (3, 2)) if dashed else "-"
        xs, ys = zip(*points[:-1]) if len(points) > 2 else ((), ())
        if len(points) > 2:
            ax.plot(xs, ys, color=INK2, linewidth=0.6, linestyle=style, solid_capstyle="butt")
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops={"arrowstyle": "-|>", "color": INK2, "linewidth": 0.6, "linestyle": style,
                                "shrinkA": 0, "shrinkB": 0, "mutation_scale": 5})

    h = 0.24
    box(0.02, 0.537, 0.38, 0.440, "Front\nimage", kind="input")
    box(0.56, 0.955, 0.94, h, "HybridNets-D3", "road + lanes")
    box(0.56, 0.638, 0.94, h, "YOLO11x-seg", "objects + masks")
    box(0.56, 0.321, 0.94, h, "Depth Anything V2", "relative depth")
    box(1.68, 1.272, 0.98, h, "Scene quality", "light, contrast, blur", "reason")
    box(1.68, 0.796, 0.98, h, "Path relation", "PATH / EDGE / SIDE", "reason")
    box(1.68, 0.480, 0.98, h, "Depth reliability", "STABLE / CHECK / UNC.", "reason")
    box(2.80, 0.704, 0.63, 0.805, "Fail-aware\nfusion\n\nrisk score,\nseverity gates", kind="fusion")
    box(2.80, 0.264, 0.63, 0.299, "Advisory +\naudit record", kind="output")
    box(0.56, 0.026, 2.10, 0.185, "Depth-only corridor fallback: CLEAR $\\rightarrow$ MONITOR", dashed=True)

    route([(0.40, 0.757), (0.56, 0.757)])
    route([(0.40, 0.854), (0.48, 0.854), (0.48, 1.074), (0.56, 1.074)])
    route([(0.40, 0.660), (0.48, 0.660), (0.48, 0.440), (0.56, 0.440)])
    route([(0.21, 0.977), (0.21, 1.390), (1.68, 1.390)])                     # image -> scene quality
    route([(1.50, 1.074), (1.59, 1.074), (1.59, 0.959), (1.68, 0.959)])       # HybridNets -> path relation
    route([(1.50, 0.801), (1.59, 0.801), (1.59, 0.871), (1.68, 0.871)])       # YOLO -> path relation
    route([(1.50, 0.713), (1.59, 0.713), (1.59, 0.642), (1.68, 0.642)])       # YOLO -> depth reliability
    route([(1.50, 0.440), (1.59, 0.440), (1.59, 0.554), (1.68, 0.554)])       # depth -> depth reliability
    route([(2.66, 1.390), (2.80, 1.390)])
    route([(2.66, 0.915), (2.80, 0.915)])
    route([(2.66, 0.598), (2.73, 0.598), (2.73, 0.774), (2.80, 0.774)])
    route([(3.115, 0.704), (3.115, 0.563)])
    route([(1.03, 0.321), (1.03, 0.211)], dashed=True)                      # depth -> fallback
    route([(2.66, 0.119), (3.115, 0.119), (3.115, 0.264)], dashed=True)     # fallback -> advisory
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(fig, "fig_pipeline")


def save(fig, name, pdf_dpi=None):
    """pdf_dpi: resolution of embedded photographs in the PDF (matplotlib resamples images to the
    save dpi, 100 by default, which blurs photos in print)."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, **({"dpi": pdf_dpi} if pdf_dpi else {}))
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[FIG] {name}")


def main():
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    fig_pipeline()
    fig_sensitivity(results)
    fig_operating(results)
    fig_qualitative()
    fig_regions()


if __name__ == "__main__":
    main()
