"""Render results.json into LaTeX tables (paper) and a Markdown digest."""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "camera_ready"
TABLES = OUT / "tables"


def pct(entry, digits=1, ci=True):
    if entry is None:
        return "--"
    value = entry["value"] if isinstance(entry, dict) else entry
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{100 * value:.{digits}f}"
    if ci and isinstance(entry, dict) and "low" in entry and not math.isnan(entry["low"]):
        text += f" [{100 * entry['low']:.{digits}f}, {100 * entry['high']:.{digits}f}]"
    return text


def integer(value):
    """Count with the paper's thousands separator (9{,}500)."""
    return f"{int(round(value)):,}".replace(",", "{,}")


def p_cell(value):
    """Same p-value format as the text macros (paper_numbers.fmt_p)."""
    if value is None:
        return "--"
    if value < 0.001:
        return "$<$0.001"
    return f"{value:.3f}" if value < 0.1 else f"{value:.2f}"


def frac(entry):
    if entry is None:
        return "--"
    return f"{integer(entry['k'])}/{integer(entry['n'])}"


def num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def table(caption, label, header, rows, align, wide=False):
    env = "table*" if wide else "table"  # table* spans both columns
    lines = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(header) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        if row == "---":
            lines.append("\\hline")
        else:
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines)


def overall(results):
    splits = results["splits"]
    cols = [s for s in ("hard", "test") if s in splits]
    if "test" not in splits:
        cols = [s for s in ("hard", "standard") if s in splits]
    header = ["Measure"] + [f"{c.capitalize()} ({integer(splits[c]['images'])})" for c in cols]

    def row(label, func):
        return [label] + [func(splits[c]) for c in cols]

    def states(s):
        st = s["reliability"]["states"]
        return "/".join(integer(st.get(state, 0)) for state in ("STABLE", "CHECK", "UNCERTAIN"))

    rows = [
        row("Accepted detections", lambda s: integer(s["reliability"]["accepted"])),
        row("\\textsc{Stable}/\\textsc{Check}/\\textsc{Uncertain}", states),
        row("Stable ratio, Wilson 95\\% CI", lambda s: pct(s["reliability"]["stable_ratio_wilson"])),
        row("Stable ratio, image-bootstrap CI", lambda s: pct(s["reliability"]["stable_ratio_cluster"])),
        "---",
        row("Detection precision (GT)", lambda s: pct(s["detection"]["precision"])),
        row("Recall, relevant road users", lambda s: pct(s["detection"]["recall_relevant"])),
        row("\\textsc{Path} precision (matched)", lambda s: pct(s["path"]["path_precision"])),
        row("\\textsc{Path} recall (matched)", lambda s: pct(s["path"]["path_recall"])),
        row("Corridor IoU with GT ego lane", lambda s: num(s["corridor"]["corridor_iou_direct"], 3)),
        "---",
        row("Occupied images", lambda s: integer(s["advisory"]["occupied_images"])),
        row("Unsafe \\textsc{Clear}, fallback on", lambda s: f"{frac(s['advisory']['unsafe_clear'])} ({pct(s['advisory']['unsafe_clear'], ci=False)})"),
        row("Unsafe \\textsc{Clear}, fallback off", lambda s: f"{frac(s['advisory_no_fallback']['unsafe_clear'])} ({pct(s['advisory_no_fallback']['unsafe_clear'], ci=False)})"),
        row("Over-caution (free ego lane)", lambda s: f"{frac(s['advisory']['over_caution'])} ({pct(s['advisory']['over_caution'], ci=False)})"),
        row("Unsupported \\textsc{Brake}/\\textsc{Stop}", lambda s: frac(s["advisory"]["severe_unsupported"]) if s["advisory"]["severe_unsupported"] else "--"),
        row("Detector-empty / fallback fired", lambda s: f"{integer(s['advisory']['no_detection_images'])}/{integer(s['advisory']['fallback_fired'])}"),
    ]
    return table(
        "Internal reliability and ground-truth results",
        "tab:overall",
        header,
        rows,
        "l" + "c" * len(cols),
    )


def train_table(results):
    """Supplementary check on the 70k BDD100K train split next to the held-out test split.

    HybridNets was trained on the train split, so its road/lane output is optimistic there; the
    no-HybridNets variant (dev road prior instead) does not use that network at all.
    """
    splits, ablations = results["splits"], results.get("ablations", {})
    columns = []
    for split, name in (("test", "Test"), ("train", "Train")):
        if split in splits:
            columns.append((f"{name} full", splits[split]))
            entry = (ablations.get(split) or {}).get("no_hybridnets")
            if entry:
                columns.append((f"{name} no-HN", entry))
    if not any(label.startswith("Train") for label, _ in columns):
        return None
    header = ["Measure"] + [f"{label} ({integer(entry['images'])})" for label, entry in columns]

    def row(label, func):
        return [label] + [func(entry) for _, entry in columns]

    def corridor(entry, field):
        value = (entry.get("corridor") or {}).get(field)
        return num(value, 3) if value is not None else "--"

    rows = [
        row("Detection precision (GT)", lambda s: pct(s["detection"]["precision"], ci=False)),
        row("Recall, relevant road users", lambda s: pct(s["detection"]["recall_relevant"], ci=False)),
        row("\\textsc{Path} precision", lambda s: pct(s["path"]["path_precision"], ci=False)),
        row("\\textsc{Path} recall", lambda s: pct(s["path"]["path_recall"], ci=False)),
        row("Road IoU (HybridNets vs GT)", lambda s: corridor(s, "road_iou_drivable")),
        row("Ego-corridor IoU", lambda s: corridor(s, "corridor_iou_direct")),
        "---",
        row("Unsafe \\textsc{Clear} (\\% occupied)", lambda s: pct(s["advisory"]["unsafe_clear"], digits=2)),
        row("Over-caution (\\% free lane)", lambda s: pct(s["advisory"]["over_caution"])),
        row("\\textsc{Brake}/\\textsc{Stop} (\\% images)", lambda s: pct(s["advisory"]["severe"], ci=False)),
    ]
    return table(
        "Supplementary check on the BDD100K train split. HybridNets was trained on these images, so its "
        "road and lane output (and the full-system path measures) are optimistic there; no-HN replaces "
        "HybridNets by the dev road prior and is free of that overlap. Rates with Wilson 95\\% CIs.",
        "tab:train",
        header,
        rows,
        "l" + "c" * len(columns),
        wide=True,  # five columns with confidence intervals do not fit one IEEE column
    )


def by_group(results, compact=False):
    """compact=True: only the test split by time of day and weather (the six-page paper); the hard-set
    categories stay in the extended version."""
    rows = []
    hard = None if compact else results["splits"].get("hard")
    if hard and hard.get("by_category"):
        rows.append(["\\multicolumn{7}{l}{\\emph{Hard set, by difficulty category}}"])
        for name in ("blur", "dark", "low_contrast", "glare", "complex"):
            entry = hard["by_category"].get(name)
            if not entry:
                continue
            rows.append(group_row(name.replace("_", " ").capitalize(), entry))
    main_split = "test" if "test" in results["splits"] else "standard"
    standard = results["splits"].get(main_split)
    if standard:
        if not compact:
            rows.append("---")
            rows.append([f"\\multicolumn{{7}}{{l}}{{\\emph{{{main_split.capitalize()} split, by time of day and weather}}}}"])
        for name in ("daytime", "dawn/dusk", "night"):
            entry = standard["by_timeofday"].get(name)
            if entry:
                rows.append(group_row(name.capitalize(), entry))
        for name, label in (("clear", "Clear"), ("rainy", "Rainy"), ("snowy", "Snowy"), ("foggy", "Foggy")):
            entry = standard["by_weather"].get(name)
            if entry and entry["images"] >= 5:
                rows.append(group_row(label, entry))
    header = ["Group", "Img.", "Obj.", "Stable", "Prec.", "Rec.", "Unsafe \\textsc{Clear}"]
    return table(
        f"{main_split.capitalize()} split by time of day and weather (\\%)" if compact
        else "Results by difficulty category, time of day, and weather (\\%)",
        "tab:groups",
        header,
        rows,
        "lcccccc",
    )


def group_row(label, entry):
    unsafe = entry["unsafe_clear"]
    unsafe_text = f"{integer(unsafe['k'])}/{integer(unsafe['n'])}" if unsafe["n"] else "--"
    return [
        label,
        integer(entry["images"]),
        integer(entry["objects"]),
        pct(entry["stable_ratio"], ci=False),
        pct(entry["precision"], ci=False),
        pct(entry["recall_relevant"], ci=False),
        unsafe_text,
    ]


BUILD_UP_ROWS = (
    ("B0", "YOLO only (road/depth priors)", "--", "--", "--", "--", "--"),
    ("B1", "+ HybridNets corridor", "--", "\\checkmark", "--", "--", "--"),
    ("B2", "+ DA V2, single region", "\\checkmark", "\\checkmark", "--", "--", "--"),
    ("B3", "+ multi-region, scene quality", "\\checkmark", "\\checkmark", "\\checkmark", "--", "--"),
    ("full", "+ gates, fallback (proposed)", "\\checkmark", "\\checkmark", "\\checkmark", "\\checkmark", "\\checkmark"),
)


def build_up(results, split="standard"):
    entries = results["ablations"].get(split, {})
    stability = results.get("stability", {})
    rows = []
    for key, label, *_ in BUILD_UP_ROWS:
        entry = entries.get(key)
        if not entry:
            continue
        adv = entry["advisory_no_fallback"] if key.startswith("B") else entry["advisory"]
        flip = stability.get(key)
        rows.append(
            [
                f"{key if key != 'full' else 'Full'}: {label}",
                f"{frac(adv['unsafe_clear'])}",
                pct(adv["unsafe_clear"], ci=False),
                pct(adv["over_caution"], ci=False),
                pct(adv["severe"], ci=False),
                pct(flip["flip_rate"][0] if flip else None, ci=False) if flip else "--",
            ]
        )
    header = ["Configuration", "Unsafe", "\\%", "Over-c.", "Severe", "Flip"]
    return table(
        f"Cumulative build-up on the {split} set (\\%). Flip: advisory change under mild perturbation (hard set)",
        f"tab:buildup_{split}",
        header,
        rows,
        "lccccc",
    )


def reliability_table(results, split="standard"):
    entry = results["splits"].get(split)
    if not entry:
        return ""
    rows = []
    for state in ("STABLE", "CHECK", "UNCERTAIN"):
        s = entry["reliability"]["by_state"][state]
        rows.append(
            [
                f"\\textsc{{{state.capitalize()}}}",
                integer(s["n"]),
                pct(s["fp_rate"]),
                pct(s["occluded_rate"], ci=False),
                pct(s["truncated_rate"], ci=False),
                num(s["mean_iou_tp"], 3),
            ]
        )
    header = ["State", "$n$", "False positive", "Occl.", "Trunc.", "IoU"]
    return table(
        f"Reliability states against BDD100K annotations ({split} set, \\%)",
        f"tab:states_{split}",
        header,
        rows,
        "lccccc",
    )


def loo_table(results, split="standard"):
    entries = results["ablations"].get(split, {})
    comparisons = results["comparisons"]
    rows = []
    for key, label in (
        ("full", "Full system"),
        ("no_hybridnets", "w/o HybridNets (road prior)"),
        ("no_depth", "w/o DA V2 (depth prior)"),
        ("single_region", "w/o multi-region (center)"),
        ("no_scene_quality", "w/o scene quality in fusion"),
        ("no_gates", "w/o severity gates"),
        ("no_fallback", "w/o depth-only fallback"),
    ):
        if key == "no_fallback":
            entry = entries.get("full")
            adv = entry["advisory_no_fallback"] if entry else None
            comp = comparisons.get(f"{split}:full_vs_no_fallback")
        else:
            entry = entries.get(key)
            adv = entry["advisory"] if entry else None
            comp = comparisons.get(f"{split}:full_vs_{key}")
        if not adv:
            continue
        p_value = comp["unsafe_clear"]["mcnemar_p"] if comp else None
        rows.append(
            [
                label,
                frac(adv["unsafe_clear"]),
                pct(adv["over_caution"], ci=False),
                pct(adv["severe"], ci=False),
                integer(comp["advisory_changes"]) if comp else "--",
                p_cell(p_value),
            ]
        )
    header = ["Variant", "Unsafe", "Over-c.", "Severe", "Changed", "$p$"]
    return table(
        f"Leave-one-out ablations on the {split} set ($p$: exact McNemar test on unsafe \\textsc{{Clear}} vs.\\ full)",
        f"tab:loo_{split}",
        header,
        rows,
        "lccccc",
    )


def sensitivity_table(results):
    rows = []
    sens = results.get("sensitivity", {})
    base = sens.get("default")
    if not base:
        return ""

    def metrics(entry):
        return (
            pct(entry["reliability"]["stable_ratio_wilson"], ci=False),
            frac(entry["advisory"]["unsafe_clear"]),
            pct(entry["advisory"]["over_caution"], ci=False),
            pct(entry["advisory"]["severe"], ci=False),
        )

    rows.append(["Published constants", *metrics(base)])
    for name, entry in sens.items():
        if name == "default":
            continue
        label = name.replace("sens:", "").replace("risk:", "risk ").replace("sq:", "SQ w=").replace("_", "\\_")
        rows.append([label, *metrics(entry)])
    header = ["Setting (dev set)", "Stable", "Unsafe", "Over-c.", "Severe"]
    return table("Threshold sensitivity on the dev split (\\%)", "tab:sensitivity", header, rows, "lcccc")


def stability_table(results):
    rows = []
    hard = results.get("stability") or {}
    std = results.get("stability_std") or {}
    for key, label in (
        ("full", "Full system"),
        ("no_gates", "w/o severity gates"),
        ("single_region", "w/o multi-region"),
        ("no_scene_quality", "w/o scene quality"),
        ("B3", "B3 (no gates, no fallback)"),
        ("B0", "B0 (YOLO only)"),
    ):
        h, s = hard.get(key), std.get(key)
        if not h and not s:
            continue
        rows.append(
            [
                label,
                pct(h["flip_rate"][0], ci=False) if h else "--",
                pct(h["severe_flip_rate"][0], ci=False) if h else "--",
                pct(s["flip_rate"][0], ci=False) if s else "--",
                pct(s["severe_flip_rate"][0], ci=False) if s else "--",
                pct(s["state_flip_rate"][0], ci=False) if s else (pct(h["state_flip_rate"][0], ci=False) if h else "--"),
            ]
        )
    header = ["Variant", "Hard flip", "Hard sev.", "1k flip", "1k sev.", "State"]
    return table(
        "Advisory stability under six mild photometric perturbations (\\%): advisory flips, severe flips, and reliability-state flips of persisting objects",
        "tab:stability",
        header,
        rows,
        "lccccc",
    )


def swap_table(results):
    rows = []
    for split in ("hard", "test"):
        for stem, label in (("yolo11s", "YOLO11s-seg"), ("yolo11m", "YOLO11m-seg"), ("full", "YOLO11x-seg")):
            entry = results["splits"].get(split) if stem == "full" else results["swap"].get(f"{split}:{stem}")
            if not entry:
                continue
            rows.append(
                [
                    f"{split.capitalize()}: {label}",
                    pct(entry["detection"]["precision"], ci=False),
                    pct(entry["detection"]["recall_relevant"], ci=False),
                    integer(entry["advisory"]["no_detection_images"]),
                    frac(entry["advisory"]["unsafe_clear"]),
                    frac(entry["advisory_no_fallback"]["unsafe_clear"]),
                ]
            )
        rows.append("---")
    if rows and rows[-1] == "---":
        rows.pop()
    header = ["Detector", "Prec.", "Rec.", "Empty", "Unsafe (FB)", "Unsafe (no FB)"]
    return table("Detector swap: the same fusion layer with three YOLO11 sizes", "tab:swap", header, rows, "lccccc")


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overall.tex": overall(results),
        "groups.tex": by_group(results),
        "groups_test.tex": by_group(results, compact=True),
        "buildup_standard.tex": build_up(results, "standard"),
        "buildup_hard.tex": build_up(results, "hard"),
        "buildup_test.tex": build_up(results, "test"),
        "states_standard.tex": reliability_table(results, "standard"),
        "states_hard.tex": reliability_table(results, "hard"),
        "states_test.tex": reliability_table(results, "test"),
        "loo_standard.tex": loo_table(results, "standard"),
        "loo_hard.tex": loo_table(results, "hard"),
        "loo_test.tex": loo_table(results, "test"),
        "sensitivity.tex": sensitivity_table(results),
        "swap.tex": swap_table(results),
        "stability.tex": stability_table(results),
        "train_check.tex": train_table(results),
    }
    outputs = {name: text for name, text in outputs.items() if text is not None}
    for name, text in outputs.items():
        (TABLES / name).write_text(text + "\n", encoding="utf-8")
    print(f"[TABLES] wrote {len(outputs)} tables to {TABLES}")


if __name__ == "__main__":
    main()
