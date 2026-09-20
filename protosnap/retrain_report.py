"""Did the promoted labels (and the target fix) help? Compares groups of checkpoints, each group
being the same recipe under different seeds, all scored as the shipped hybrid on the frozen
validation and test sets.

    python -m protosnap.retrain_report \\
        --group "Stage 4 model (old targets, original data)" checkpoints/unet_xc/best.pt \\
        --group "Original data" checkpoints/retrain/base_s0/best.pt checkpoints/retrain/base_s1/best.pt \\
        --group "Original + promoted" checkpoints/retrain/plus_s0/best.pt checkpoints/retrain/plus_s1/best.pt

Without human Esagil labels to score against, two proxies are measured on the still-unreviewed
glyphs: the share of wedges whose tail leaves the ink (model-independent; on labelled data an
off-ink tail is almost always a real error) and the share the model itself marks doubtful.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .analysis import failure_classes
from .data import find_pairs, load_skeleton
from .imageprep import prepare_image
from .metrics import evaluate
from .split import load_splits, pairs_in_split

DOUBTFUL = 0.3


def _summ(a) -> dict:
    pe = a.point_error
    return {"precision": a.precision, "recall": a.recall, "f1": a.f1, "corner_error": a.corner_error, "apex": pe["apex"],
            "heads": (pe["head_a"] + pe["head_b"]) / 2, "tail": pe["tail"], "exact": a.exact_rate}


def unreviewed_proxy(hybrid, root: Path, review_csv: Path, per_font: int, seed: int = 0) -> dict:
    with open(review_csv, newline="") as f:
        open_rows = [r for r in csv.DictReader(f) if not r["accept"].strip() and r.get("source", "model") == "model"]
    rng = np.random.default_rng(seed)
    out = {}
    for font in sorted({r["font"] for r in open_rows}):
        rows = [r for r in open_rows if r["font"] == font]
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=min(per_font, len(rows)), replace=False))]
        n = off = doubt = 0
        for r in rows:
            image, _ = prepare_image(next((root / "prototypes" / font).glob(f"*/{r['codepoint']}.png")))
            sw = hybrid.predict_scored(image, None)
            n += len(sw)
            off += sum(s.off_ink for s in sw)
            doubt += sum(s.confidence < DOUBTFUL for s in sw)
        out[font] = {"glyphs": len(rows), "wedges": n, "off_ink_rate": off / max(n, 1), "doubtful_rate": doubt / max(n, 1)}
    return out


def main(argv=None) -> int:
    from .hybrid import HybridPredictor
    from .learned.predictor import LearnedPredictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", nargs="+", required=True, metavar=("LABEL", "CHECKPOINT"))
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--review", default="reports/stage5/review/review_all.csv")
    ap.add_argument("--proxy-glyphs", type=int, default=120, help="unreviewed glyphs per font for the proxy measures")
    ap.add_argument("--out", default="reports/stage5/retrain")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(root)
    sk = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    sp = {n: pairs_in_split(pairs, splits, n) for n in ("val", "test")}

    result = {"groups": []}
    for g in args.group:
        label, ckpts = g[0], g[1:]
        runs = []
        for ck in ckpts:
            hy = HybridPredictor(LearnedPredictor.from_checkpoint(ck))
            run = {"checkpoint": ck}
            for s in ("val", "test"):
                r = evaluate(hy, sp[s], s, skeletons=sk)
                run[s] = {"all": _summ(r.overall), **{f: _summ(a) for f, a in r.per_font.items()}}
            run["failures_test"] = dict(failure_classes(hy, sp["test"], sk))
            run["proxy"] = unreviewed_proxy(hy, root, Path(args.review), args.proxy_glyphs)
            runs.append(run)
            t = run["test"]["all"]
            print(f"{label:46s} {Path(ck).parent.name:10s} test F1 {t['f1']:.3f} err {t['corner_error']:.2f} tail {t['tail']:.2f} exact {t['exact']:.3f}", flush=True)
        result["groups"].append({"label": label, "runs": runs})
    with open(out / "comparison.json", "w") as f:
        json.dump(result, f, indent=1)
    write_markdown(result, out / "comparison.md")
    try:
        from .charts import chart_retrain
        chart_retrain(result, out.parent.parent / "charts")
    except Exception as e:
        print(f"chart skipped: {e}")
    print(f"report -> {out / 'comparison.md'}")
    return 0


def _mean(runs, *path) -> float:
    vals = []
    for r in runs:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def _spread(runs, *path) -> str:
    vals = []
    for r in runs:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return "" if len(vals) < 2 else f" ({min(vals):.3f} to {max(vals):.3f})"


def write_markdown(result: dict, path: Path) -> None:
    L = ["# Retraining comparison", "",
         "Every checkpoint is scored as the shipped hybrid on the frozen validation and test sets (original human annotations only). "
         "Values are means over seeds with the range in brackets. One glyph is 2.2 points of exact-match on a 45-glyph split, so "
         "differences smaller than the seed range are not evidence.", "",
         "![retrain](../../charts/stage5_retrain.png)", ""]
    for split in ("test", "val"):
        L += [f"## {split.capitalize()} split", "", "| Model | Seeds | F1 | Corner err (px) | Heads | Tail | Exact-match |", "|---|---|---|---|---|---|---|"]
        for g in result["groups"]:
            r = g["runs"]
            L.append(f"| {g['label']} | {len(r)} | {_mean(r, split, 'all', 'f1'):.3f}{_spread(r, split, 'all', 'f1')} | {_mean(r, split, 'all', 'corner_error'):.2f}"
                     f"{_spread(r, split, 'all', 'corner_error')} | {_mean(r, split, 'all', 'heads'):.2f} | {_mean(r, split, 'all', 'tail'):.2f}{_spread(r, split, 'all', 'tail')} "
                     f"| {_mean(r, split, 'all', 'exact'):.3f}{_spread(r, split, 'all', 'exact')} |")
        L.append("")
    L += ["## Test split per font", "", "| Model | Font | F1 | Corner err | Tail | Exact-match |", "|---|---|---|---|---|---|"]
    for g in result["groups"]:
        for font in ("Assurbanipal", "Santakku"):
            r = g["runs"]
            L.append(f"| {g['label']} | {font} | {_mean(r, 'test', font, 'f1'):.3f} | {_mean(r, 'test', font, 'corner_error'):.2f} | {_mean(r, 'test', font, 'tail'):.2f} | {_mean(r, 'test', font, 'exact'):.3f} |")
    classes = list(result["groups"][0]["runs"][0]["failures_test"])
    L += ["", "## Failure classes on test (mean count over seeds)", "", "| Failure class | " + " | ".join(g["label"] for g in result["groups"]) + " |",
          "|---|" + "---|" * len(result["groups"])]
    for c in classes:
        L.append(f"| {c} | " + " | ".join(f"{np.mean([r['failures_test'][c] for r in g['runs']]):.1f}" for g in result["groups"]) + " |")
    fonts = sorted(result["groups"][0]["runs"][0]["proxy"])
    L += ["", "## Proxies on still-unreviewed glyphs (no ground truth)", "",
          "Off-ink tails are a model-independent validity check; doubtful wedges are the model's own confidence below 0.3. Lower is better for both.", "",
          "| Model | " + " | ".join(f"{f}: off-ink tails | {f}: doubtful wedges" for f in fonts) + " |", "|---|" + "---|---|" * len(fonts)]
    for g in result["groups"]:
        L.append(f"| {g['label']} | " + " | ".join(f"{_mean(g['runs'], 'proxy', f, 'off_ink_rate'):.1%} | {_mean(g['runs'], 'proxy', f, 'doubtful_rate'):.1%}" for f in fonts) + " |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
