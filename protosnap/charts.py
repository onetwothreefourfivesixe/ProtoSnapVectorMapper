"""Bar-chart figures for the model comparison, drawn from the saved result JSON.

    python -m protosnap.charts [--reports reports] [--out reports/charts] [--split test]

Figures
  metrics_<split>.png          eight small multiples: four rates and four pixel errors, one bar per model
  metrics_by_font_<split>.png  F1, exact-match and corner error per font
  failure_classes_<split>.png  large-error counts per class, classical versus the selected learned model

Design notes. Colour encodes model identity in a fixed order (validated categorical palette:
blue, orange, aqua, yellow; the Stage 1 jittered oracle is a neutral grey reference, not a
model). Rates and pixel errors never share an axis; each metric gets its own panel and the
pixel panels share one scale so apex, head and tail errors compare directly. Every bar is
direct-labelled with its value and the rows are named, so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REFERENCE_GREY = "#b5b3ab"
# fixed model -> colour assignment (colour follows the entity, never its rank)
MODELS = [
    ("classical", "Classical baseline", "#2a78d6"),
    ("unet", "U-Net", "#eb6834"),
    ("unet_xc", "U-Net + classical channels", "#1baf7a"),
    ("resnet34", "ResNet-34 encoder", "#eda100"),
    ("hybrid", "Hybrid: U-Net + channels, tails refined by geometry", "#4a3aa7"),
]
ROW_LABEL = {"hybrid": "Hybrid (tails refined)", "jittered_oracle": "Jittered oracle (reference)"}   # short names for row ticks
STAGE3_KEYS = ("classical", "unet", "unet_xc", "resnet34")
STAGE4_KEYS = ("classical", "unet_xc", "hybrid")     # validated adjacent order: blue, aqua, violet
REFERENCE = ("jittered_oracle", "Jittered oracle (2 px noise, reference)", REFERENCE_GREY)


def _style() -> None:
    for f in Path("/mnt/c/Windows/Fonts").glob("segoeui*.ttf"):
        try:
            font_manager.fontManager.addfont(str(f))
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })


def _load(reports: Path, split: str) -> dict[str, dict]:
    out = {}
    paths = {"classical": reports / "stage2" / "json" / f"classical_{split}.json",
             "jittered_oracle": reports / "stage1" / "json" / f"jittered_oracle_{split}.json"}
    for key in ("unet", "unet_xc", "resnet34"):
        paths[key] = reports / "stage3" / key / f"{split}.json"
    paths["hybrid"] = reports / "stage4" / "json" / f"hybrid_{split}.json"
    for key, p in paths.items():
        if p.exists():
            with open(p) as f:
                out[key] = json.load(f)
    return out


def _bar(ax, y: float, value: float, height: float, color: str, radius_px: float = 4.0) -> None:
    """Horizontal bar: square at the baseline, rounded at the data end."""
    if value <= 0:
        return
    (x0p, y0p), (x1p, y1p) = ax.transData.transform([(0, 0), (1, 1)])
    rx = radius_px / abs(x1p - x0p)
    ry = radius_px / abs(y1p - y0p)
    rx, ry = min(rx, value), min(ry, height / 2)
    b, t = y - height / 2, y + height / 2
    k = 0.5523
    verts = [(0, b), (value - rx, b), (value - rx + k * rx, b), (value, b + ry - k * ry), (value, b + ry),
             (value, t - ry), (value, t - ry + k * ry), (value - rx + k * rx, t), (value - rx, t), (0, t), (0, b)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def _panel(ax, xmax: float, ticks: list[float], fmt: str) -> None:
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks], fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(axis="both", length=0)


def _legend(fig, entries: list[tuple[str, str]], inches_from_top: float = 0.86) -> None:
    y = 1 - inches_from_top / fig.get_figheight()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for _, c in entries]
    fig.legend(handles, [n for n, _ in entries], loc="upper left", bbox_to_anchor=(0.012, y), ncol=len(entries),
               frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.0, columnspacing=1.6, labelcolor=INK2)


def _titles(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.18 / h, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 1 - 0.52 / h, subtitle, fontsize=10, color=INK2, va="top")


def _value(ax, x: float, y: float, text: str, bold: bool = False) -> None:
    """Value label at a bar tip, in ink (never the series colour), backed by the surface so a
    gridline never strikes through the digits."""
    ax.text(x, y, text, va="center", ha="left", fontsize=9, color=INK if bold else INK2,
            fontweight="bold" if bold else "normal", zorder=4,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


# --------------------------------------------------------------------------- figure A

def chart_metrics(data: dict[str, dict], split: str, out: Path, keys=STAGE3_KEYS, prefix: str = "") -> Path:
    series = [m for m in MODELS if m[0] in data and m[0] in keys] + ([REFERENCE] if REFERENCE[0] in data else [])
    o = {k: data[k]["overall"] for k, _, _ in series}
    heads = lambda a: (a["point_error"]["head_a"] + a["point_error"]["head_b"]) / 2
    rows = [
        [("F1", lambda a: a["f1"]), ("Precision", lambda a: a["precision"]), ("Recall", lambda a: a["recall"]),
         ("Exact-match (glyphs solved)", lambda a: a["exact_rate"])],
        [("Mean corner error", lambda a: a["corner_error"]), ("Apex error", lambda a: a["point_error"]["apex"]),
         ("Head tip error", heads), ("Tail end error", lambda a: a["point_error"]["tail"])],
    ]
    px_max = max(f(o[k]) for _, f in rows[1] for k, _, _ in series) * 1.22
    n = len(series)
    fig = plt.figure(figsize=(13.2, 6.6), dpi=120)
    g = data[series[0][0]]["overall"]
    _titles(fig, "Model comparison on the held-out test split" if split == "test" else f"Model comparison on the {split} split",
            f"{g['n_glyphs']} glyphs, {g['n_gt']} ground-truth wedges, match threshold 10 px. Top row: higher is better. Bottom row: pixels, lower is better.")
    _legend(fig, [(name, c) for _, name, c in series])
    left, width, gap = 0.205, 0.178, 0.018
    for r, row in enumerate(rows):
        bottom = 0.47 if r == 0 else 0.06
        for c, (title, fn) in enumerate(row):
            ax = fig.add_axes([left + c * (width + gap), bottom, width, 0.30])
            ax.set_ylim(n - 0.5, -0.5)
            if r == 0:
                _panel(ax, 1.18, [0, 0.5, 1.0], "{:.1f}")
            else:
                _panel(ax, px_max, [t for t in (0, 2, 4, 6, 8) if t < px_max], "{:.0f} px")
            ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
            vals = [fn(o[k]) for k, _, _ in series]
            model_vals = [v for v, (k, _, _) in zip(vals, series) if k != REFERENCE[0]]
            best = max(model_vals) if r == 0 else min(model_vals)
            fig.canvas.draw()
            for i, ((k, name, col), v) in enumerate(zip(series, vals)):
                _bar(ax, i, v, 0.56, col)
                is_best = k != REFERENCE[0] and abs(v - best) < 1e-12
                _value(ax, v + ax.get_xlim()[1] * 0.02, i, f"{v:.3f}" if r == 0 else f"{v:.2f}", is_best)
            if c == 0:
                ax.set_yticks(range(n))
                ax.set_yticklabels([ROW_LABEL.get(k, name) for k, name, _ in series], fontsize=9.5)
            else:
                ax.set_yticks([])
    fig.text(0.015, 0.012, "Bold value = best model in the panel. The grey bar is the Stage 1 reference predictor, not a model.",
             fontsize=8.5, color=MUTED)
    path = out / f"{prefix}metrics_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- figure B

def chart_by_font(data: dict[str, dict], split: str, out: Path, keys=STAGE3_KEYS, prefix: str = "") -> Path:
    series = [m for m in MODELS if m[0] in data and m[0] in keys]
    fonts = sorted(data[series[0][0]]["per_font"])
    panels = [("F1", lambda a: a["f1"], True), ("Exact-match (glyphs solved)", lambda a: a["exact_rate"], True),
              ("Mean corner error", lambda a: a["corner_error"], False)]
    n = len(series)
    fig = plt.figure(figsize=(13.2, 5.2), dpi=120)
    counts = ", ".join(f"{f} {data[series[0][0]]['per_font'][f]['n_glyphs']} glyphs" for f in fonts)
    _titles(fig, f"Per-font results on the {split} split", f"{counts}. Every later stage is required to hold up on both fonts, not only on the pooled split.")
    _legend(fig, [(name, c) for _, name, c in series])
    left, width, gap = 0.105, 0.27, 0.03
    slot = n + 1.2
    for c, (title, fn, rate) in enumerate(panels):
        ax = fig.add_axes([left + c * (width + gap), 0.09, width, 0.60])
        ax.set_ylim(len(fonts) * slot - 1.0, -0.8)
        vmax = 1.18 if rate else max(fn(data[k]["per_font"][f]) for k, _, _ in series for f in fonts) * 1.2
        _panel(ax, vmax, [0, 0.5, 1.0] if rate else [t for t in (0, 1, 2, 3, 4, 5) if t < vmax], "{:.1f}" if rate else "{:.0f} px")
        ax.set_title(title + ("" if rate else " (lower is better)"), fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
        fig.canvas.draw()
        for fi, font in enumerate(fonts):
            vals = [fn(data[k]["per_font"][font]) for k, _, _ in series]
            best = max(vals) if rate else min(vals)
            for i, ((k, name, col), v) in enumerate(zip(series, vals)):
                y = fi * slot + i
                _bar(ax, y, v, 0.62, col)
                is_best = abs(v - best) < 1e-12
                _value(ax, v + vmax * 0.015, y, f"{v:.3f}" if rate else f"{v:.2f}", is_best)
        if c == 0:
            ax.set_yticks([fi * slot + (n - 1) / 2 for fi in range(len(fonts))])
            ax.set_yticklabels(fonts, fontsize=10.5, color=INK)
        else:
            ax.set_yticks([])
    fig.text(0.015, 0.015, "Bold value = best model for that font. Bars within a font are in legend order.", fontsize=8.5, color=MUTED)
    path = out / f"{prefix}metrics_by_font_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- figure C

def chart_failures(fc: dict, split: str, out: Path, prefix: str = "") -> Path | None:
    names = [n for n in fc["counts"]]
    if len(names) < 2:
        return None
    label = {k: n for k, n, _ in MODELS}
    colour = {k: c for k, _, c in MODELS}
    total = lambda c: sum(fc["counts"][n][c] for n in names)
    classes = [c for c in sorted(fc["counts"][names[0]], key=lambda c: -total(c)) if total(c) > 0]
    vmax = max(max(fc["counts"][n].values()) for n in names) * 1.15
    k = len(names)
    H = (0.26 * k + 0.26) * len(classes) + 2.0
    fig = plt.figure(figsize=(11.0, H), dpi=120)
    _titles(fig, f"Where the errors are: failure classes on the {split} split",
            "Counts of wedges (or unmatched detections) per class. Same definitions for every model, from protosnap.analysis.")
    _legend(fig, [(label.get(n, n), colour.get(n, REFERENCE_GREY)) for n in names])
    ax = fig.add_axes([0.37, 0.45 / H, 0.59, (H - 1.45 - 0.45) / H])
    slot = k + 1.0
    ax.set_ylim(len(classes) * slot - 0.9, -1.0)
    _panel(ax, vmax, list(range(0, int(vmax) + 1, 5 if vmax <= 40 else 10)), "{:.0f}")
    fig.canvas.draw()
    for ci, cls in enumerate(classes):
        for i, m in enumerate(names):
            v = fc["counts"][m][cls]
            _bar(ax, ci * slot + i, v, 0.78, colour.get(m, REFERENCE_GREY))
            _value(ax, v + vmax * 0.012, ci * slot + i, str(v))
    ax.set_yticks([ci * slot + (k - 1) / 2 for ci in range(len(classes))])
    ax.set_yticklabels([c[0].upper() + c[1:] for c in classes], fontsize=9.5, color=INK)
    path = out / f"{prefix}failure_classes_{split}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_confidence(rows: list[dict], out: Path) -> Path | None:
    """Stage 5: how confident the model is on the unlabelled glyphs, one panel per font. A count
    per confidence bin is a magnitude, so every bar is the same single hue (no legend needed)."""
    _style()
    out.mkdir(parents=True, exist_ok=True)
    fonts = [f for f in sorted({r["font"] for r in rows}) if any(r["font"] == f and r["source"] == "model" for r in rows)]
    if not fonts:
        return None
    import numpy as np
    edges = np.linspace(0, 1, 11)
    hist = {f: np.histogram([float(r["confidence"]) for r in rows if r["font"] == f and r["source"] == "model"], bins=edges)[0] for f in fonts}
    fig = plt.figure(figsize=(13.2, 4.3), dpi=120)
    _titles(fig, "Model confidence on the unlabelled glyphs",
            "Glyph confidence is its least certain wedge. Low bins are the review queue. Esagil is a font the model never trained on.")
    width, gap, left = 0.86 / len(fonts) - 0.03, 0.045, 0.06
    for i, f in enumerate(fonts):
        ax = fig.add_axes([left + i * (width + gap), 0.17, width, 0.55])
        ymax = max(1, hist[f].max()) * 1.18
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="both", length=0, labelsize=8.5)
        n = int(hist[f].sum())
        ax.set_title(f"{f}  ({n} glyphs)", fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
        ax.set_xlabel("glyph confidence", fontsize=9, color=INK2)
        if i == 0:
            ax.set_ylabel("glyphs", fontsize=9, color=INK2)
        fig.canvas.draw()
        for b, v in enumerate(hist[f]):
            _column(ax, edges[b] + 0.008, edges[b + 1] - 0.008, float(v), MODELS[0][2])
        k = int(hist[f].argmax())
        ax.text((edges[k] + edges[k + 1]) / 2, hist[f][k] + ymax * 0.02, str(int(hist[f][k])), ha="center", va="bottom", fontsize=9, color=INK2)
    path = out / "stage5_confidence.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _column(ax, x0: float, x1: float, value: float, color: str, radius_px: float = 4.0) -> None:
    """Vertical bar: square at the baseline, rounded cap."""
    if value <= 0:
        return
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (1, 1)])
    rx, ry = radius_px / abs(ax1 - ax0), radius_px / abs(ay1 - ay0)
    rx, ry = min(rx, (x1 - x0) / 2), min(ry, value)
    k = 0.5523
    verts = [(x0, 0), (x0, value - ry), (x0, value - ry + k * ry), (x0 + rx - k * rx, value), (x0 + rx, value),
             (x1 - rx, value), (x1 - rx + k * rx, value), (x1, value - ry + k * ry), (x1, value - ry), (x1, 0), (x0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def chart_retrain(result: dict, out: Path) -> Path:
    """Retraining comparison: one bar per model group (mean over seeds), with each seed drawn as a
    dot so the seed-to-seed spread is visible next to the difference between groups."""
    _style()
    out.mkdir(parents=True, exist_ok=True)
    groups = result["groups"]
    colours = [REFERENCE_GREY, MODELS[0][2], MODELS[1][2], MODELS[2][2]]      # grey = earlier reference model, then slots 1-3
    colours = colours[1:] if len(groups) < 3 else colours
    panels = [("F1", "f1", True), ("Exact-match (glyphs solved)", "exact", True), ("Mean corner error", "corner_error", False), ("Tail end error", "tail", False)]
    n = len(groups)
    fig = plt.figure(figsize=(13.2, 1.9 + 0.62 * n), dpi=120)
    H = fig.get_figheight()
    _titles(fig, "Retraining on reviewed labels: held-out test split",
            "Bars are means over seeds, dots are the individual seeds. A gap smaller than the dot spread is not evidence of a difference.")
    _legend(fig, [(g["label"], colours[i % len(colours)]) for i, g in enumerate(groups)])
    left, width, gap = 0.04, 0.215, 0.025
    px_max = max(r["test"]["all"][k] for g in groups for r in g["runs"] for _, k, rate in panels if not rate) * 1.25
    for c, (title, key, rate) in enumerate(panels):
        ax = fig.add_axes([left + c * (width + gap), 0.42 / H, width, (H - 1.55 - 0.42) / H])
        ax.set_ylim(n - 0.5, -0.5)
        _panel(ax, 1.18 if rate else px_max, [0, 0.5, 1.0] if rate else [t for t in (0, 1, 2, 3, 4, 5, 6) if t < px_max], "{:.1f}" if rate else "{:.0f} px")
        ax.set_title(title + ("" if rate else " (lower is better)"), fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
        ax.set_yticks([])
        fig.canvas.draw()
        means = [sum(r["test"]["all"][key] for r in g["runs"]) / len(g["runs"]) for g in groups]
        best = max(means) if rate else min(means)
        for i, (g, m) in enumerate(zip(groups, means)):
            _bar(ax, i, m, 0.56, colours[i % len(colours)])
            vals = [r["test"]["all"][key] for r in g["runs"]]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [i, i], color=INK, linewidth=2, zorder=4, solid_capstyle="round")   # seed range
                ax.scatter(vals, [i] * len(vals), s=42, color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
            _value(ax, max(vals) + ax.get_xlim()[1] * 0.045, i, f"{m:.3f}" if rate else f"{m:.2f}", abs(m - best) < 1e-12)
    path = out / "stage5_retrain.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(reports: Path, out: Path, split: str = "test") -> list[Path]:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    data = _load(reports, split)
    made = []
    if "classical" in data and any(k in data for k in ("unet", "unet_xc", "resnet34")):
        made += [chart_metrics(data, split, out), chart_by_font(data, split, out)]
    if "classical" in data and "hybrid" in data:
        made += [chart_metrics(data, split, out, STAGE4_KEYS, "stage4_"), chart_by_font(data, split, out, STAGE4_KEYS, "stage4_")]
    for stage, prefix in (("stage3", ""), ("stage4", "stage4_")):
        fc_path = reports / stage / f"failure_classes_{split}.json"
        if fc_path.exists():
            with open(fc_path) as f:
                p = chart_failures(json.load(f), split, out, prefix)
            if p:
                made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="reports/charts")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args(argv)
    for p in make_all(Path(args.reports), Path(args.out), args.split):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
