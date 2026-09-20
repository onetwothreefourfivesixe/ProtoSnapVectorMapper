"""Stage 1 runner: score the reference predictors (null, oracle, jittered oracle) on every
split, write reports/stage1/stats.md and JSON results, and render a failure gallery.

    python -m protosnap.stage1 [--root .] [--out reports/stage1] [--splits data/splits.json] [--threshold 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import find_pairs, load_image, load_metadata, load_skeleton
from .metrics import (DEFAULT_MATCH_THRESHOLD, EvalResult, JitteredOraclePredictor, NullPredictor,
                      OraclePredictor, evaluate, markdown_table)
from .render import comparison_panel, save_sheets
from .split import SPLIT_NAMES, load_splits, pairs_in_split


def failure_gallery(result: EvalResult, pairs_by_key, skeletons, predictor, out_dir: Path, stem: str,
                    k: int = 24, meta=None) -> list[Path]:
    cells = []
    for g in result.worst(k):
        p = pairs_by_key[g.key]
        img = load_image(p)
        gt = skeletons[g.key].wedges
        pred = predictor.predict(img, p)
        from .metrics import match_wedges
        matches, up, ug = match_wedges(pred, gt, result.match_threshold)
        panel = comparison_panel(img, gt, pred, matches, up, ug, scale=1)
        name = (meta or {}).get(p.codepoint, {}).get("name", "")
        err = "nan" if g.corner_error != g.corner_error else f"{g.corner_error:.1f}"
        cap = f"{p.font} {p.codepoint} {name[:18]}\nGT {g.n_gt} pred {g.n_pred} FP {g.fp} FN {g.fn} err {err}px"
        cells.append((panel, cap))
    if not cells:
        return []
    return save_sheets(cells, out_dir, stem, per_page=12, cols=3, title=f"Worst glyphs: {result.predictor} on {result.split}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/stage1")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(root)
    by_key = {p.key: p for p in pairs}
    skeletons = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    meta = load_metadata(root)

    predictors = [NullPredictor(), OraclePredictor(skeletons),
                  JitteredOraclePredictor(sigma=2.0, drop=0.1, spurious=0.05, skeletons=skeletons)]
    results: list[EvalResult] = []
    for name in SPLIT_NAMES:
        sp = pairs_in_split(pairs, splits, name)
        for pr in predictors:
            if hasattr(pr, "rng"):
                pr.rng = __import__("numpy").random.default_rng(0)
            r = evaluate(pr, sp, split=name, match_threshold=args.threshold, skeletons=skeletons)
            r.to_json(out / "json" / f"{pr.name.split('(')[0]}_{name}.json")
            results.append(r)

    lines = ["# Stage 1 report: evaluation harness", "",
             f"Match threshold: {args.threshold:g} px on apex distance. Corner error is the mean distance over the 4 wedge points of matched wedges. "
             "Exact = no FP, no FN, every matched point within the threshold. Only wedge strokes are evaluated; triangle and malformed strokes are excluded from ground truth.",
             "", "## Reference predictors", "", markdown_table(results), "",
             "Reading the table: `null` is the floor (recall 0). `oracle` is the ceiling and must show precision, recall, F1 and exact all equal to 1 with zero error. "
             "`jittered_oracle` perturbs the ground truth by 2 px Gaussian noise per point, drops 10% of wedges and adds 5% spurious ones; "
             "it shows what the numbers look like for a model that is nearly right, and its gallery below demonstrates the failure renderer.", ""]
    (out / "stats.md").write_text("\n".join(lines) + "\n")

    if not args.no_gallery:
        jit = [r for r in results if r.split == "test" and r.predictor.startswith("jittered")][0]
        pr = predictors[2]
        pr.rng = __import__("numpy").random.default_rng(0)
        # re-evaluate with a fresh rng so gallery predictions match the scored ones
        jit = evaluate(pr, pairs_in_split(pairs, splits, "test"), "test", args.threshold, skeletons)
        pr.rng = __import__("numpy").random.default_rng(0)
        pages = failure_gallery(jit, by_key, skeletons, pr, out / "gallery", "jittered_oracle_test", meta=meta)
        print(f"gallery pages: {[str(p) for p in pages]}")

    print(f"report -> {out / 'stats.md'}")
    for r in results:
        a = r.overall
        print(f"{r.predictor:48s} {r.split:5s} P={a.precision:.3f} R={a.recall:.3f} F1={a.f1:.3f} err={a.corner_error:.2f} exact={a.exact_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
