"""Stage 4 runner: hybrid refinement, ablation, confidence evaluation and ship decision.

    python -m protosnap.stage4 [--checkpoint checkpoints/unet_xc/best.pt] [--out reports/stage4]

Writes reports/stage4/stats.md, json/ (per-split results of the hybrid), failure_classes_test.json,
confidence_test.json, the failure gallery, and the stage4_* charts under reports/charts/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import failure_classes, failure_table
from .classical import ClassicalPredictor
from .data import find_pairs, load_image, load_metadata, load_skeleton
from .hybrid import HybridPredictor, glyph_confidence
from .metrics import DEFAULT_MATCH_THRESHOLD, evaluate, match_wedges
from .split import SPLIT_NAMES, load_splits, pairs_in_split
from .stage1 import failure_gallery

IGNORE_RADIUS = 14.0   # px; same radius the training loss ignores around non-wedge strokes


def _objective(a) -> float:
    err = a.corner_error if a.corner_error == a.corner_error else 50.0
    return a.f1 + a.exact_rate - err / 100.0


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def confidence_report(hybrid: HybridPredictor, pairs, skeletons, correct_px: float = 5.0) -> dict:
    wedge_conf, wedge_apex, wedge_ok, glyph_conf, glyph_ok = [], [], [], [], []
    for p in pairs:
        sw = hybrid.predict_scored(load_image(p), p)
        pred = [s.wedge for s in sw]
        m, up, ug = match_wedges(pred, skeletons[p.key].wedges)
        ok = {x.pred_index: max(x.point_errors) <= correct_px for x in m}
        for i, s in enumerate(sw):
            wedge_conf.append(s.confidence)
            wedge_apex.append(s.apex_score)
            wedge_ok.append(ok.get(i, False))
        glyph_conf.append(glyph_confidence(sw))
        glyph_ok.append(not up and not ug and all(ok.get(i, False) for i in range(len(sw))))
    order = np.argsort(glyph_conf)[::-1]
    gok = np.array(glyph_ok)[order]
    triage = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(frac * len(gok))))
        triage.append({"most_confident_fraction": frac, "glyphs": k, "fully_correct_rate": float(gok[:k].mean())})
    return {"correct_px": correct_px, "wedges": len(wedge_ok), "wedge_correct_rate": float(np.mean(wedge_ok)),
            "wedge_auroc_confidence": auroc(wedge_conf, wedge_ok), "wedge_auroc_apex_score_only": auroc(wedge_apex, wedge_ok),
            "glyphs": len(glyph_ok), "glyph_correct_rate": float(np.mean(glyph_ok)),
            "glyph_auroc_confidence": auroc(glyph_conf, glyph_ok), "triage": triage}


def _row(label: str, split: str, a, note: str = "") -> str:
    pe = a.point_error
    return (f"| {label} | {split} | {a.precision:.3f} | {a.recall:.3f} | {a.f1:.3f} | {a.corner_error:.2f} | {pe['apex']:.2f} "
            f"| {(pe['head_a'] + pe['head_b']) / 2:.2f} | {pe['tail']:.2f} | {a.exact_rate:.3f} | {note} |")


HEADER = ["| Configuration | Split | Precision | Recall | F1 | Corner err | Apex | Heads | Tail | Exact | Note |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]


def main(argv=None) -> int:
    from .learned.predictor import LearnedPredictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--checkpoint", default="checkpoints/unet_xc/best.pt")
    ap.add_argument("--out", default="reports/stage4")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--gallery-size", type=int, default=24)
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(root)
    by_key = {p.key: p for p in pairs}
    sk = {p.key: load_skeleton(p) for p in pairs}
    splits = load_splits(args.splits)
    meta = load_metadata(root)
    sp = {n: pairs_in_split(pairs, splits, n) for n in SPLIT_NAMES}

    classical = ClassicalPredictor()
    learned = LearnedPredictor.from_checkpoint(args.checkpoint)
    hybrid = HybridPredictor(learned)
    hybrid.name = "hybrid"
    configs = [
        ("Classical baseline (Stage 2)", classical, ""),
        (f"Learned: {learned.name} (Stage 3)", learned, "proposer"),
        ("Hybrid, ink check only", hybrid.with_params(refine_tail=False), ""),
        ("Hybrid, tail to nearest traced node", hybrid.with_params(tail_mode="nearest"), "naive refinement"),
        ("Hybrid, tail by heatmap vote along the trace", hybrid, "**shipped configuration**"),
        ("  + snap head tips to skeleton endpoints", hybrid.with_params(snap_tips=True), ""),
        ("  + orientation swap rule", hybrid.with_params(fix_orientation=True), ""),
    ]
    lines = ["# Stage 4 report: hybrid refinement", "",
             f"Proposer: `{args.checkpoint}`. Match threshold {DEFAULT_MATCH_THRESHOLD:g} px, frozen Stage 1 metric unless stated. "
             "Hybrid settings were chosen on the validation split; test is reported once.", "", "## Ablation", ""] + HEADER
    res = {}
    for label, pr, note in configs:
        for split in ("val", "test"):
            r = evaluate(pr, sp[split], split, skeletons=sk)
            res[(label, split)] = r
            lines.append(_row(label, split, r.overall, note if split == "test" else ""))

    # ---- per font, shipped configuration vs its two parents
    shipped = "Hybrid, tail by heatmap vote along the trace"
    lines += ["", "## Per font on test", "", "| Font | Model | F1 | Corner err | Tail | Exact |", "|---|---|---|---|---|---|"]
    for font in sorted(res[(shipped, "test")].per_font):
        for label in (configs[0][0], configs[1][0], shipped):
            a = res[(label, "test")].per_font[font]
            lines.append(f"| {font} | {label} | {a.f1:.3f} | {a.corner_error:.2f} | {a.point_error['tail']:.2f} | {a.exact_rate:.3f} |")

    # ---- secondary metric
    lines += ["", f"## Secondary metric: predictions within {IGNORE_RADIUS:g} px of a non-wedge stroke are skipped", "",
              "Tail-less triangles and the malformed stroke are real heads without a wedge label, so a detection there is "
              "unscoreable rather than wrong. The training loss already ignores those regions. This table is informational; "
              "the frozen metric above remains the headline.", ""] + HEADER
    for label, pr, _ in (configs[0], configs[1], configs[4]):
        r = evaluate(pr, sp["test"], "test", skeletons=sk, ignore_non_wedge=IGNORE_RADIUS)
        lines.append(_row(label, "test", r.overall))

    # ---- save hybrid results, failure classes, confidence
    for split in SPLIT_NAMES:
        r = res.get((shipped, split)) or evaluate(hybrid, sp[split], split, skeletons=sk)
        r.to_json(out / "json" / f"hybrid_{split}.json")
    fc = {"classical": failure_classes(classical, sp["test"], sk), learned.name: failure_classes(learned, sp["test"], sk),
          "hybrid": failure_classes(hybrid, sp["test"], sk)}
    with open(out / "failure_classes_test.json", "w") as f:
        json.dump({"split": "test", "counts": {k: dict(v) for k, v in fc.items()}}, f, indent=1)
    lines += ["", "## Failure classes on test", "", failure_table(fc)]

    conf = {s: confidence_report(hybrid, sp[s], sk) for s in ("val", "test")}
    with open(out / "confidence.json", "w") as f:
        json.dump(conf, f, indent=1)
    lines += ["", "## Confidence", "",
              "Per-wedge confidence is the weakest of the network's apex score and its heatmap support at the tail end and both head tips, "
              "times 0.6 when the tail points in a rare direction. A glyph's confidence is the minimum over its wedges. "
              f"A wedge counts as correct when matched with every point within {conf['test']['correct_px']:g} px.", "",
              "| Split | Wedge AUROC, confidence | Wedge AUROC, apex score only | Glyph AUROC, confidence |", "|---|---|---|---|"]
    for s in ("val", "test"):
        c = conf[s]
        lines.append(f"| {s} | {c['wedge_auroc_confidence']:.3f} | {c['wedge_auroc_apex_score_only']:.3f} | {c['glyph_auroc_confidence']:.3f} |")
    lines += ["", "Triage on test: sort glyphs by confidence and keep the most confident fraction.", "",
              "| Most confident | Glyphs | Fully correct within 5 px |", "|---|---|---|"]
    for t in conf["test"]["triage"]:
        lines.append(f"| {t['most_confident_fraction']:.0%} | {t['glyphs']} | {t['fully_correct_rate']:.3f} |")

    # ---- ship decision on validation
    lv, hv = res[(configs[1][0], "val")].overall, res[(shipped, "val")].overall
    lt, ht = res[(configs[1][0], "test")].overall, res[(shipped, "test")].overall
    decision = "hybrid" if _objective(hv) > _objective(lv) else "learned"
    lines += ["", "## Ship decision", "",
              f"On validation the hybrid scores objective {_objective(hv):.3f} against {_objective(lv):.3f} for the learned model alone "
              f"(exact-match {hv.exact_rate:.3f} vs {lv.exact_rate:.3f}, tail error {hv.point_error['tail']:.2f} vs {lv.point_error['tail']:.2f} px). "
              f"On test: exact-match {ht.exact_rate:.3f} vs {lt.exact_rate:.3f}, tail error {ht.point_error['tail']:.2f} vs {lt.point_error['tail']:.2f} px, "
              f"corner error {ht.corner_error:.2f} vs {lt.corner_error:.2f} px. Recommended configuration: **{decision}**.", ""]

    pages = failure_gallery(res[(shipped, "test")], by_key, sk, hybrid, out / "gallery", "hybrid_test", k=args.gallery_size, meta=meta)
    from .charts import make_all
    charts = [c for c in make_all(out.parent, out.parent / "charts", "test") if c.name.startswith("stage4_")]
    lines[4:4] = ["## Charts", ""] + [f"![{c.stem}](../charts/{c.name})\n" for c in charts]
    lines += [f"Failure gallery of the hybrid on test: " + ", ".join(f"`{pg}`" for pg in pages), ""]
    (out / "stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("|") and ("test" in l or "val" in l))[:6000])
    print(f"\ndecision: {decision}\nreport -> {out / 'stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
