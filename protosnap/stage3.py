"""Stage 3 runner: score every trained checkpoint against the classical baseline, write
reports/stage3/stats.md, and render the failure gallery of the best learned model.

    python -m protosnap.stage3 [--checkpoints checkpoints] [--out reports/stage3]

Train models first with `python -m protosnap.learned.train --name <run>`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import failure_classes, failure_table
from .classical import ClassicalPredictor
from .data import find_pairs, load_metadata, load_skeleton
from .metrics import DEFAULT_MATCH_THRESHOLD, evaluate, markdown_table
from .split import load_splits, pairs_in_split
from .stage1 import failure_gallery


def _objective(a) -> float:
    err = a.corner_error if a.corner_error == a.corner_error else 50.0
    return a.f1 + a.exact_rate - err / 100.0


def main(argv=None) -> int:
    from .learned.predictor import LearnedPredictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--out", default="reports/stage3")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--gallery-size", type=int, default=24)
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(root)
    by_key = {p.key: p for p in pairs}
    skeletons = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    meta = load_metadata(root)
    val, test = pairs_in_split(pairs, splits, "val"), pairs_in_split(pairs, splits, "test")

    predictors = [ClassicalPredictor()]
    infos = {}
    for ck in sorted(Path(args.checkpoints).glob("*/best.pt")):
        p = LearnedPredictor.from_checkpoint(ck)
        predictors.append(p)
        log = out / p.name / "log.json"
        infos[p.name] = json.load(open(log)) if log.exists() else {}
    if len(predictors) == 1:
        print("no checkpoints found; train a model first")
        return 1

    results = {}
    for p in predictors:
        results[p.name] = {"val": evaluate(p, val, "val", skeletons=skeletons), "test": evaluate(p, test, "test", skeletons=skeletons)}

    learned = [p for p in predictors if p.name != "classical"]
    best = max(learned, key=lambda p: _objective(results[p.name]["val"].overall))
    pages = failure_gallery(results[best.name]["test"], by_key, skeletons, best, out / "gallery", f"{best.name}_test",
                            k=args.gallery_size, meta=meta)

    fc = {"classical": failure_classes(predictors[0], test, skeletons), best.name: failure_classes(best, test, skeletons)}
    with open(out / "failure_classes_test.json", "w") as f:
        json.dump({"split": "test", "selected": best.name, "counts": {k: dict(v) for k, v in fc.items()}}, f, indent=1)
    from .charts import make_all
    charts = make_all(out.parent, out.parent / "charts", "test")   # classical results come from reports/stage2/json

    lines = ["# Stage 3 report: learned keypoint model", "",
             f"Match threshold {DEFAULT_MATCH_THRESHOLD:g} px. Metric definitions as in Stage 1. Model selection and decode "
             "tuning use the validation split only; the test split is scored once per checkpoint.", "",
             "## Runs", "", "| Run | Encoder | Extra channels | Params (M) | Epochs | Best epoch | Train time (min) | Decode |", "|---|---|---|---|---|---|---|---|"]
    for p in learned:
        i = infos.get(p.name, {})
        a = i.get("args", {})
        lines.append(f"| {p.name} | {a.get('encoder', '?')} | {a.get('extra_channels', '?')} | {i.get('params', 0) / 1e6:.1f} | {a.get('epochs', '?')} "
                     f"| {i.get('best_epoch', '?')} | {i.get('train_seconds', 0) / 60:.1f} | thr {p.cfg.threshold}, snap {p.cfg.snap_radius} |")
    lines += ["", "## Charts", ""] + [f"![{c.stem}](../charts/{c.name})\n" for c in charts]
    lines += ["", "## Scores", "", markdown_table([results[p.name][s] for s in ("val", "test") for p in predictors]), ""]

    lines += [f"## Review gate: `{best.name}` (best on validation) versus the classical baseline on test", "",
              "| Font | Classical F1 | Learned F1 | Classical err | Learned err | Classical exact | Learned exact | Learned better? |", "|---|---|---|---|---|---|---|---|"]
    c, l = results["classical"]["test"], results[best.name]["test"]
    verdicts = []
    for font in ["all"] + sorted(c.per_font):
        ca = c.overall if font == "all" else c.per_font[font]
        la = l.overall if font == "all" else l.per_font[font]
        better = _objective(la) > _objective(ca)
        if font != "all":
            verdicts.append(better)
        lines.append(f"| {font} | {ca.f1:.3f} | {la.f1:.3f} | {ca.corner_error:.2f} | {la.corner_error:.2f} | {ca.exact_rate:.3f} | {la.exact_rate:.3f} | {'yes' if better else 'NO'} |")
    gate = "PASSED" if all(verdicts) else "NOT PASSED"
    lines += ["", f"Gate (learned beats classical on the combined objective for both fonts): **{gate}**.", "",
              "## Failure classes on test", "", failure_table(fc), "",
              f"Curves: " + ", ".join(f"`{out / p.name / 'curves.png'}`" for p in learned),
              f"Failure gallery of `{best.name}` on test: " + ", ".join(f"`{pg}`" for pg in pages), ""]
    (out / "stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))
    print(f"report -> {out / 'stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
