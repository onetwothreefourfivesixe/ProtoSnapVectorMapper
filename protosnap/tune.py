"""Coordinate-descent tuning of ClassicalParams.

Objective (maximised): F1 + exact_rate - corner_error / 100.

The classical model never fits on the training split, so by default the search uses train
plus val (404 glyphs) and the test split stays untouched. Tuning on val alone (45 glyphs)
let single-wedge differences flip decisions that clearly hurt on test.

    python -m protosnap.tune [--rounds 3] [--tune-split train+val] [--init params.json] [--out reports/stage2/tuned_params.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .classical import ClassicalParams, ClassicalPredictor
from .data import find_pairs, load_skeleton
from .metrics import evaluate
from .split import load_splits, pairs_in_split

GRID: dict[str, list] = {
    "max_hole_area": [16, 24, 40, 60],
    "head_radius": [3.0, 4.0, 5.0, 6.0],
    "head_radius_scale": [0.8, 1.0, 1.3, 1.6],
    "side_max_len": [12.0, 16.0, 20.0, 28.0],
    "opening_weight": [0.0, 0.25, 0.5, 1.0],
    "tail_max_dev": [45.0, 60.0, 90.0],
    "no_tail_penalty": [15.0, 30.0, 60.0],
    "missing_prong_penalty": [0.0, 8.0, 20.0],
    "capped_prong_penalty": [10.0, 25.0, 50.0],
    "prong_max_dev": [50.0, 70.0, 90.0],
    "prong_dev_weight": [0.0, 0.5, 1.0, 2.0],
    "prong_dir_from_centroid": [True, False],
    "tail_dir_weight": [0.0, 0.5, 1.0, 2.0, 4.0],
    "prong_dir_weight": [0.0, 0.5, 1.0, 2.0, 4.0],
    "prong_max_len": [16.0, 20.0, 24.0],
    "prong_default_len": [6.0, 9.0, 12.0],
    "tip_extension": [0.0, 1.0, 2.0],
    "apex_shift": [-1.0, 0.0, 1.0, 2.0],
    "continue_angle": [25.0, 35.0, 45.0, 60.0],
    "max_tail_hops": [1, 2, 4, 8],
    "max_prong_hops": [0, 1, 2],
    "stop_at_heads": [True, False],
    "pass_through_heads": [True, False],
    "merge_angle": [15.0, 30.0, 45.0, 60.0],
    "nms_apex_dist": [3.0, 6.0, 8.0, 10.0],
    "nms_angle": [30.0, 45.0, 60.0, 90.0],
}


def objective(pred: ClassicalPredictor, pairs, skeletons) -> tuple[float, dict]:
    a = evaluate(pred, pairs, "tune", skeletons=skeletons).overall
    err = a.corner_error if a.corner_error == a.corner_error else 50.0
    return a.f1 + a.exact_rate - err / 100.0, {"f1": a.f1, "precision": a.precision, "recall": a.recall,
                                               "corner_error": err, "exact": a.exact_rate}


def tune(base: ClassicalParams, pairs, skeletons, rounds: int = 2, verbose: bool = True) -> tuple[ClassicalParams, dict]:
    pred = ClassicalPredictor(base)
    best_score, best_stats = objective(pred, pairs, skeletons)
    params = base
    for r in range(rounds):
        improved = False
        for name, values in GRID.items():
            for v in values:
                if getattr(params, name) == v:
                    continue
                cand = replace(params, **{name: v})
                score, stats = objective(pred.with_params(**asdict(cand)), pairs, skeletons)
                if score > best_score + 1e-9:
                    best_score, best_stats, params, improved = score, stats, cand, True
                    if verbose:
                        print(f"round {r + 1}: {name}={v!r} -> score {score:.4f} f1 {stats['f1']:.3f} err {stats['corner_error']:.2f} exact {stats['exact']:.3f}")
        if not improved:
            break
    return params, {"score": best_score, **best_stats}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="reports/stage2/tuned_params.json")
    ap.add_argument("--init", default=None, help="warm-start from a tuned_params.json instead of the defaults")
    ap.add_argument("--tune-split", default="train+val", help="split(s) to tune on, joined with +")
    args = ap.parse_args(argv)
    pairs = find_pairs(args.root)
    sk = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    val = [p for name in args.tune_split.split("+") for p in pairs_in_split(pairs, splits, name)]
    base = ClassicalParams()
    if args.init:
        with open(args.init) as f:
            known = {k: v for k, v in json.load(f)["params"].items() if hasattr(base, k)}
        base = replace(base, **known)
    params, stats = tune(base, val, sk, args.rounds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"params": asdict(params), "tune_split": args.tune_split, "tune_stats": stats}, f, indent=1)
    print(json.dumps(asdict(params), indent=1))
    print("val:", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
