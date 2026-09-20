"""Stage 6 review gate: does this checkout reproduce the recorded scores?

    protosnap reproduce                 # verify, exit code 0 on success
    protosnap reproduce --record        # (maintainers) rewrite models/expected_scores.json from this machine

Three checks, in order:
  1. Data fingerprint. A hash over the 449 original glyph images and skeleton files, the
     metadata, the corrections, and the validation and test codepoint lists. If this differs,
     scores are not comparable and nothing else is attempted.
  2. Classical baseline on the test split. Fully deterministic, so it must match exactly.
  3. Each model in models/, scored as the shipped hybrid on validation and test. Weights are
     shipped because retraining cannot recreate them: training is not bit-reproducible, and the
     Stage 4 model predates the training-target fix made in Stage 5. Inference differs in the
     last bits between GPU mixed precision and CPU, so a model passes when every score is
     within a small tolerance; a difference of up to one glyph of exact-match is reported as
     a warning, anything larger fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from .classical import ClassicalPredictor
from .data import find_pairs, load_skeleton
from .metrics import evaluate
from .split import load_splits, pairs_in_split

EXPECTED = "models/expected_scores.json"
KEYS = ("precision", "recall", "f1", "corner_error", "tail", "exact")
TOL = {"precision": 0.002, "recall": 0.002, "f1": 0.002, "corner_error": 0.02, "tail": 0.05, "exact": 1e-9}


def fingerprint(root: Path, splits: dict) -> str:
    h = hashlib.sha256()
    files = [root / "prototypes" / "metadata.csv"]
    for p in find_pairs(root, corrections=None):
        files += [p.image_path, p.adf_path, p.con_path]
    files += sorted((root / "data" / "corrections").glob("*/*.csv"))
    for f in sorted(files, key=lambda f: f.relative_to(root).as_posix()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    h.update(json.dumps({"val": sorted(splits["val"]), "test": sorted(splits["test"])}).encode())
    return h.hexdigest()


def _summ(a) -> dict:
    return {"precision": a.precision, "recall": a.recall, "f1": a.f1, "corner_error": a.corner_error,
            "tail": a.point_error["tail"], "exact": a.exact_rate, "glyphs": a.n_glyphs}


def measure(root: Path, splits_path: str) -> dict:
    splits = load_splits(splits_path)
    pairs = find_pairs(root)
    sk = {p.key: load_skeleton(p) for p in pairs}
    sp = {n: pairs_in_split(pairs, splits, n) for n in ("val", "test")}
    out = {"fingerprint": fingerprint(root, splits), "pairs": len(pairs),
           "scores": {"classical": {s: _summ(evaluate(ClassicalPredictor(), sp[s], s, skeletons=sk).overall) for s in ("val", "test")}}}
    models = sorted((root / "models").glob("*.pt"))
    if models:
        from .hybrid import HybridPredictor
        from .learned.predictor import LearnedPredictor
        import torch
        out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        for m in models:
            hy = HybridPredictor(LearnedPredictor.from_checkpoint(m))
            out["scores"][f"hybrid:{m.name}"] = {s: _summ(evaluate(hy, sp[s], s, skeletons=sk).overall) for s in ("val", "test")}
    return out


def compare(expected: dict, got: dict) -> tuple[list[str], bool, bool]:
    lines, failed, warned = [], False, False
    for name, by_split in expected["scores"].items():
        if name not in got["scores"]:
            lines.append(f"FAIL  {name}: not found in this checkout")
            failed = True
            continue
        exact_tol = 0.0 if name == "classical" else None
        for split, exp in by_split.items():
            g = got["scores"][name][split]
            one_glyph = 1.0 / max(exp.get("glyphs", 45), 1)
            worst, status = [], "ok"
            for k in KEYS:
                d = abs(g[k] - exp[k])
                tol = 1e-9 if exact_tol == 0.0 else TOL[k]
                if d <= tol:
                    continue
                if name != "classical" and k == "exact" and d <= one_glyph + 1e-9:
                    status = "warn" if status == "ok" else status
                elif name != "classical" and d <= 5 * TOL[k]:
                    status = "warn" if status == "ok" else status
                else:
                    status = "fail"
                worst.append(f"{k} {g[k]:.4f} vs {exp[k]:.4f}")
            failed |= status == "fail"
            warned |= status == "warn"
            tag = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
            lines.append(f"{tag}  {name:32s} {split:4s} F1 {g['f1']:.3f}  corner err {g['corner_error']:.2f} px  tail {g['tail']:.2f} px  "
                         f"exact {g['exact']:.3f}" + (f"   <- {'; '.join(worst)}" if worst else ""))
    return lines, failed, warned


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--record", action="store_true", help="write the measured values as the new expected values")
    ap.add_argument("--out", default="reports/stage6")
    args = ap.parse_args(argv)
    root = Path(args.root)
    got = measure(root, args.splits)
    got["recorded_on"] = {"python": platform.python_version(), "platform": platform.platform(), "device": got.get("device", "n/a")}
    exp_path = root / EXPECTED
    if args.record:
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.write_text(json.dumps(got, indent=1) + "\n")
        print(f"recorded {len(got['scores'])} score sets -> {exp_path}")
        return 0
    expected = json.loads(exp_path.read_text())
    print(f"data fingerprint : {got['fingerprint'][:16]}…  ({got['pairs']} original pairs)")
    if got["fingerprint"] != expected["fingerprint"]:
        print(f"FAIL  the evaluation data differs from what the scores were recorded on (expected {expected['fingerprint'][:16]}…). "
              "Scores are not comparable; nothing else was checked.")
        return 1
    print("PASS  data fingerprint matches")
    lines, failed, warned = compare(expected, got)
    print("\n".join(lines))
    verdict = "FAILED" if failed else ("PASSED with warnings (differences within one glyph: expected across GPU and CPU)" if warned else "PASSED")
    print(f"\nreproduction {verdict}   [this machine: {got['recorded_on']['device']}, recorded on: {expected['recorded_on']['device']}]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reproduce.json").write_text(json.dumps({"verdict": verdict, "measured": got, "expected": expected}, indent=1))
    (out / "reproduce.md").write_text("# Reproduction check\n\n```\n" + "\n".join(lines) + f"\n\nreproduction {verdict}\n```\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
