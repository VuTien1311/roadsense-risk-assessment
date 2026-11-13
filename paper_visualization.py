"""Original | overlay | relative-depth presentation; no model inference.

The raw Depth Anything output is used once, after inference. Its higher raw
values indicate nearer structure in this pipeline. Colors are normalized per
image and must never be interpreted as meters or compared across images.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Overlay rendering -------------------------------------------------------------------------
# Presentation only: everything drawn here is read from the finished fusion output. Colors (RGB)
# follow the advisory of each object, from neutral to warm; the road tint is the HybridNets
# drivable area and the yellow marks are its lane pixels.
ADVISORY_ORDER = ("CLEAR", "MONITOR", "SLOW", "BRAKE", "STOP")
ADVISORY_RGB = {
    "CLEAR": (226, 232, 240),
    "MONITOR": (56, 189, 248),
    "SLOW": (250, 204, 21),
    "BRAKE": (251, 146, 60),
    "STOP": (239, 68, 68),
}
ROAD_RGB, LANE_RGB = (34, 197, 94), (253, 224, 71)
CARD_RGBA, TEXT_RGB, MUTED_RGB, WARN_RGB = (15, 23, 42, 214), (248, 250, 252), (148, 163, 184), (252, 211, 77)
FULL_LABEL_LIMIT = 8  # more labels than this hide the scene; the other objects get a one-letter tag


def _font(size):
    try:
        return ImageFont.load_default(size=max(8, int(round(size))))
    except TypeError:  # Pillow without a scalable default font
        return ImageFont.load_default()


def _text_size(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, top


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _blend(image, mask, color, alpha):
    if mask is None:
        return
    mask = np.asarray(mask)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    selected = mask > 0
    if selected.any():
        image[selected] = (image[selected] * (1.0 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def _object_label(det, compact=False):
    """(main text, depth text, depth is uncertain) for one fused detection."""
    if compact:
        main = f"{det.get('class', 'object')} {det.get('corridor_relation', '')}".rstrip()
    else:
        main = f"{det.get('class', 'object')} {float(det.get('confidence', 0.0)):.2f}  {det.get('corridor_relation', '')}".rstrip()
    unit = det.get("original_inverse_depth_unit")
    state = str(det.get("original_unit_status", "")).upper()
    uncertain = state != "STABLE"
    depth = f"u{float(unit):.1f}" if unit is not None else "u n/a"
    if uncertain:
        depth += f" {state or 'UNKNOWN'}"  # the depth state is written out when it is not STABLE
    return main, depth, uncertain


def render_overlay(image, road_mask=None, lane_mask=None, detections=(), advisory="CLEAR", advisory_confidence=None,
                   critical=None, scene_quality=None, depth_scan=None, text_scale=1.0, max_labels=FULL_LABEL_LIMIT,
                   compact=False):
    """Draw the fused result on the input image (BGR uint8 in, BGR uint8 out, same size).

    detections: the per-object audit records of the report (class, confidence, box_xyxy,
    corridor_relation, original_unit_status, original_inverse_depth_unit, advisory, risk_score).
    Nothing is computed here: the overlay only shows what the fusion layer decided. ``u`` is the
    relative depth index (smaller = nearer within this image, never meters); when the depth estimate
    is not STABLE, its state (CHECK or UNCERTAIN) is written after u.

    text_scale, max_labels and compact only change the presentation (larger text and fewer labels for
    a small printed figure; compact keeps the advisory card to the advisory and the scene state and
    leaves the color legend to the figure caption).
    """
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or 0 in frame.shape[:2]:
        raise ValueError("image must be a non-empty uint8 HxWx3 BGR array.")
    height, width = frame.shape[:2]
    scale = max(0.55, width / 1280.0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _blend(rgb, road_mask, ROAD_RGB, 0.30)
    _blend(rgb, lane_mask, LANE_RGB, 0.70)
    if road_mask is not None and np.asarray(road_mask).shape[:2] == (height, width):
        contours, _ = cv2.findContours((np.asarray(road_mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edged = rgb.copy()
        cv2.drawContours(edged, contours, -1, ROAD_RGB, max(1, int(round(1.5 * scale))), cv2.LINE_AA)
        rgb = cv2.addWeighted(edged, 0.6, rgb, 0.4, 0)

    canvas = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    label_font, tag_font = _font(15 * scale * text_scale), _font(12 * scale * text_scale)
    pad, radius, stripe = int(round(5 * scale * text_scale)), int(round(5 * scale)), max(3, int(round(4 * scale * text_scale)))
    line = max(2, int(round(2 * scale * max(1.0, text_scale * 0.75))))

    objects = []
    for det in detections or ():
        box = det.get("box_xyxy") or det.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width - 1)))
        y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height - 1)))
        action = str(det.get("advisory", "CLEAR")).upper()
        severity = ADVISORY_ORDER.index(action) if action in ADVISORY_ORDER else 0
        objects.append({"det": det, "box": (x1, y1, x2, y2), "color": ADVISORY_RGB[ADVISORY_ORDER[severity]],
                        "severity": severity, "risk": float(det.get("risk_score") or 0.0)})

    # Far objects first, so nearer boxes are drawn over them.
    for item in sorted(objects, key=lambda o: o["box"][3]):
        x1, y1, x2, y2 = item["box"]
        if item["severity"] >= 2:
            draw.rectangle((x1, y1, x2, y2), fill=item["color"] + (46,))
        draw.rectangle((x1, y1, x2, y2), outline=(15, 23, 42, 230), width=line + 2)
        draw.rectangle((x1, y1, x2, y2), outline=item["color"] + (255,), width=line)

    # Header card: the global advisory and the evidence behind it.
    advisory = str(advisory).upper()
    advisory_color = ADVISORY_RGB.get(advisory, ADVISORY_RGB["CLEAR"])
    big, small = _font(30 * scale * text_scale), _font(14 * scale * text_scale)
    lines = []
    if compact:
        if scene_quality:
            lines.append((f"scene {scene_quality.get('status', '?')}", MUTED_RGB))
    elif critical:
        lines.append((str(critical), TEXT_RGB))
    facts = []
    if scene_quality and not compact:
        flags = ",".join(scene_quality.get("flags") or [])
        facts.append(f"Scene {scene_quality.get('status', '?')} {float(scene_quality.get('score', 0.0)):.2f}" + (f" ({flags})" if flags else ""))
    if advisory_confidence is not None:
        facts.append(f"Evidence {float(advisory_confidence):.2f}")
    if not compact:
        facts.append(f"Objects {len(objects)}")
        lines.append(("   ".join(facts), MUTED_RGB))
        if depth_scan and depth_scan.get("status"):
            lines.append((f"Depth-only path {depth_scan['status']}", MUTED_RGB))
        lines.append(("u = relative depth index (smaller = nearer, not meters)", MUTED_RGB))
    margin, inner = int(round(12 * scale)), int(round(10 * scale * text_scale))
    caption_w, caption_h, _ = _text_size(draw, "ADVISORY", small)
    word_w, word_h, word_top = _text_size(draw, advisory, big)
    sizes = [_text_size(draw, text, small) for text, _ in lines]
    row = int(round(caption_h * 1.55))
    card_w = max([caption_w + inner + word_w] + [w for w, _, _ in sizes]) + 2 * inner + stripe
    card_h = 2 * inner + max(word_h, caption_h) + int(round(6 * scale)) + row * len(lines)
    card = (margin, margin, min(width - margin, margin + card_w), min(height - margin, margin + card_h))
    draw.rounded_rectangle(card, radius=2 * radius, fill=CARD_RGBA)
    draw.rounded_rectangle((card[0], card[1], card[0] + stripe, card[3]), radius=stripe // 2, fill=advisory_color + (255,))
    text_x, text_y = card[0] + stripe + inner, card[1] + inner
    draw.text((text_x, text_y + max(0, word_h - caption_h)), "ADVISORY", font=small, fill=MUTED_RGB + (255,))
    draw.text((text_x + caption_w + inner, text_y - word_top), advisory, font=big, fill=advisory_color + (255,))
    text_y += max(word_h, caption_h) + int(round(6 * scale))
    for text, color in lines:
        draw.text((text_x, text_y), text, font=small, fill=color + (255,))
        text_y += row

    # Legend, bottom left (left to the caption in compact mode).
    entries = [(name, ADVISORY_RGB[name]) for name in ADVISORY_ORDER] + [("road", ROAD_RGB), ("lane", LANE_RGB)]
    swatch = int(round(10 * scale))
    widths = [_text_size(draw, name, tag_font)[0] for name, _ in entries]
    legend_w = sum(widths) + len(entries) * (swatch + int(round(14 * scale))) + inner
    legend_h = int(round(22 * scale))
    legend = (margin, height - margin - legend_h, min(width - margin, margin + legend_w), height - margin)
    if compact:
        entries, legend = [], (0, 0, 0, 0)
    else:
        draw.rounded_rectangle(legend, radius=radius, fill=CARD_RGBA)
    cursor = legend[0] + inner
    for (name, color), text_w in zip(entries, widths):
        top = legend[1] + (legend_h - swatch) // 2
        draw.rectangle((cursor, top, cursor + swatch, top + swatch), fill=color + (255,))
        draw.text((cursor + swatch + int(round(4 * scale)), legend[1] + int(round(4 * scale))), name, font=tag_font, fill=TEXT_RGB + (255,))
        cursor += swatch + int(round(14 * scale)) + text_w

    # Labels: the most severe objects get a full label, placed where it covers no other label.
    taken = [card, legend]
    ranked = sorted(objects, key=lambda o: (-o["severity"], -o["risk"], -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))
    full = 0
    for item in ranked:
        x1, y1, x2, y2 = item["box"]
        main, depth, uncertain = _object_label(item["det"], compact)
        main_w, main_h, main_top = _text_size(draw, main + "  ", label_font)
        depth_w, depth_h, _ = _text_size(draw, depth, label_font)
        label_w, label_h = stripe + 2 * pad + main_w + depth_w, max(main_h, depth_h) + 2 * pad
        spot = None
        if full < max_labels and (y2 - y1) >= 0.03 * height:
            for left, top in ((x1, y1 - label_h - 2), (x1, y2 + 2), (x2 - label_w, y1 - label_h - 2), (x2 - label_w, y2 + 2), (x1 + line, y1 + line)):
                left = min(max(left, 2), max(2, width - label_w - 2))
                top = min(max(top, 2), max(2, height - label_h - 2))
                candidate = (left, top, left + label_w, top + label_h)
                if not any(_overlaps(candidate, other) for other in taken):
                    spot = candidate
                    break
        if spot is not None:
            full += 1
            taken.append(spot)
            draw.rounded_rectangle(spot, radius=radius, fill=CARD_RGBA)
            draw.rounded_rectangle((spot[0], spot[1], spot[0] + stripe, spot[3]), radius=stripe // 2, fill=item["color"] + (255,))
            base_x, base_y = spot[0] + stripe + pad, spot[1] + pad - main_top
            draw.text((base_x, base_y), main, font=label_font, fill=TEXT_RGB + (255,))
            draw.text((base_x + main_w, base_y), depth, font=label_font, fill=(WARN_RGB if uncertain else TEXT_RGB) + (255,))
            continue
        if compact:
            continue  # small printed figure: the colored box alone shows the advisory
        tag = str(item["det"].get("corridor_relation", "")) or "?"
        tag_w, tag_h, tag_top = _text_size(draw, tag, tag_font)
        left, top = min(x1, width - tag_w - 6), max(0, y1 - tag_h - 5)
        box = (left, top, left + tag_w + 6, top + tag_h + 5)
        if any(_overlaps(box, other) for other in taken):
            continue  # the colored box still shows the advisory; a tag here would cover a label
        taken.append(box)
        draw.rectangle(box, fill=item["color"] + (255,))
        draw.text((left + 3, top + 2 - tag_top), tag, font=tag_font, fill=(15, 23, 42, 255))

    result = np.asarray(Image.alpha_composite(canvas, layer).convert("RGB"))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def colorize_depth(raw_depth):
    values = np.asarray(raw_depth, dtype=np.float32)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Depth must be a non-empty HxW map.")
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(values[valid], [2, 98])
        if float(high - low) > 1e-6:
            normalized[valid] = np.clip((values[valid] - low) / (high - low), 0, 1)
        else:
            normalized[valid] = 0.5
    else:
        low = high = None
    view = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    view[~valid] = (24, 24, 24)
    info = {
        "normalization": "per-image positive finite raw depth, percentiles 2/98",
        "percentile_2": float(low) if low is not None else None,
        "percentile_98": float(high) if high is not None else None,
        "valid_fraction": float(valid.mean()),
        "meaning": "Warm colors: relatively nearer; cool colors: relatively farther. Not meters.",
        "invalid_color_bgr": [24, 24, 24],
    }
    return view, info


def make_comparison(original, overlay, depth_view, panel_width=640):
    frames = [np.asarray(image) for image in (original, overlay, depth_view)]
    for image in frames:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Each panel must be a uint8 HxWx3 BGR image.")
        if 0 in image.shape[:2]:
            raise ValueError("Panel images cannot be empty.")
    if any(image.shape != frames[0].shape for image in frames[1:]):
        raise ValueError("Original, overlay and depth must have matching source geometry.")
    if panel_width < 320:
        raise ValueError("panel_width must be at least 320.")

    source_h, source_w = frames[0].shape[:2]
    panel_height = max(1, round(source_h * panel_width / source_w))
    header, footer, gap = 52, 84, 12
    canvas = np.full((header + panel_height + footer, 3 * panel_width + 4 * gap, 3), 20, np.uint8)
    headings = ("ORIGINAL", "OVERLAY", "RELATIVE DEPTH")
    captions = ("Unmodified input image", "Objects, road/lane and advisory", "Per-image depth colors, not meters")
    texts = []  # (x, y, text, size, RGB), drawn in one anti-aliased pass at the end
    for index, image in enumerate(frames):
        left = gap + index * (panel_width + gap)
        resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA if panel_width < source_w else cv2.INTER_LINEAR)
        canvas[header:header + panel_height, left:left + panel_width] = resized
        texts.append((left + 4, 14, headings[index], 24, (235, 235, 235)))
        texts.append((left + 4, header + panel_height + 10, captions[index], 15, (195, 195, 195)))

    depth_left = gap + 2 * (panel_width + gap)
    bar_width = panel_width - 16
    gradient = np.tile(np.linspace(0, 255, bar_width).astype(np.uint8), (10, 1))
    legend = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    bar_y = header + panel_height + 34
    canvas[bar_y:bar_y + 10, depth_left + 8:depth_left + 8 + bar_width] = legend
    text_y = header + panel_height + 52
    texts.append((depth_left + 8, text_y, "FAR", 14, (215, 215, 215)))
    texts.append((depth_left + panel_width - 48, text_y, "NEAR", 14, (215, 215, 215)))
    texts.append((gap + 4, text_y, "Static-image research output; no vehicle control.", 14, (165, 165, 165)))
    picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    for x, y, text, size, color in texts:
        draw.text((x, y), text, font=_font(size), fill=color)
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def write_png(path, image):
    path = Path(path)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed to encode image: {path}")
    # tofile handles Unicode paths on Windows, unlike some cv2.imwrite builds.
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Failed to write image: {path}")


def save_visualizations(original, overlay, raw_depth, output_dir, stem):
    depth_view, normalization = colorize_depth(raw_depth)
    if depth_view.shape != original.shape:
        raise ValueError("Raw depth must match the input image resolution.")
    comparison = make_comparison(original, overlay, depth_view)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "original": output / f"{stem}_original.png",
        "overlay": output / f"{stem}_overlay.png",
        "relative_depth": output / f"{stem}_relative_depth.png",
        "comparison": output / f"{stem}_comparison.png",
        "raw_depth": output / f"{stem}_raw_depth.npy",
    }
    for key, frame in (("original", original), ("overlay", overlay), ("relative_depth", depth_view), ("comparison", comparison)):
        write_png(files[key], frame)
    np.save(str(files["raw_depth"]), np.asarray(raw_depth), allow_pickle=False)
    return {"files": {key: str(path.resolve()) for key, path in files.items()}, "depth_display": normalization, "layout": "original | overlay | relative depth"}
