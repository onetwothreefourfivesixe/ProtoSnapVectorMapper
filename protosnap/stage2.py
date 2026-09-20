"""Stage 2 runner: evaluate the tuned classical baseline on every split, write
reports/stage2/stats.md and JSON, and render the failure gallery on the test split.

    python -m protosnap.stage2 [--params reports/stage2/tuned_params.json] [--out reports/stage2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classical import ClassicalParams, ClassicalPredictor
from .data import find_pairs, load_metadata, load_skeleton
from .metrics import DEFAULT_MATCH_THRESHOLD, evaluate, markdown_table
from .split import SPLIT_NAMES, load_splits, pairs_in_split
from .stage1 import failure_gallery


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/stage2")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--params", default="reports/stage2/tuned_params.json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    ap.add_argument("--gallery-size", type=int, default=24)
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(root)
    by_key = {p.key: p for p in pairs}
    skeletons = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    meta = load_metadata(root)

    params_path = Path(args.params)
    if params_path.exists():
        with open(params_path) as f:
            params = ClassicalParams(**json.load(f)["params"])
        source = f"tuned on val (`{params_path}`)"
    else:
        params = ClassicalParams()
        source = "defaults (no tuned_params.json found)"
    pred = ClassicalPredictor(params)

    results = []
    for name in SPLIT_NAMES:
        r = evaluate(pred, pairs_in_split(pairs, splits, name), split=name, match_threshold=args.threshold, skeletons=skeletons)
        r.to_json(out / "json" / f"classical_{name}.json")
        results.append(r)

    test = [r for r in results if r.split == "test"][0]
    pages = failure_gallery(test, by_key, skeletons, pred, out / "gallery", "classical_test", k=args.gallery_size, meta=meta)

    lines = ["# Stage 2 report: classical baseline", "",
             f"Parameters: {source}. Match threshold {args.threshold:g} px. Metric definitions as in Stage 1.", "",
             "```", json.dumps(params.__dict__, indent=1), "```", "",
             "## Scores", "", markdown_table(results), "",
             f"Failure gallery (worst {args.gallery_size} test glyphs): " + ", ".join(f"`{p}`" for p in pages), ""]
    (out / "stats.md").write_text("\n".join(lines) + "\n")
    for r in results:
        a = r.overall
        print(f"{r.split:5s} P={a.precision:.3f} R={a.recall:.3f} F1={a.f1:.3f} err={a.corner_error:.2f} "
              f"apex={a.point_error['apex']:.2f} heads={(a.point_error['head_a']+a.point_error['head_b'])/2:.2f} tail={a.point_error['tail']:.2f} exact={a.exact_rate:.3f}")
    print(f"report -> {out / 'stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
