"""Stage 5: batch inference, export, confidence ranking and the review loop.

    python -m protosnap.stage5 run        [--checkpoint checkpoints/unet_xc/best.pt] [--out generated] [--report reports/stage5]
    python -m protosnap.stage5 acceptance reports/stage5/review/review_random.csv
    python -m protosnap.stage5 promote    reports/stage5/review/review_random.csv [--corrected data/reviewed] [--dry-run]

`run` writes machine skeletons to generated/<Font>/<hex>_adf.csv and _con.csv in the same
format as skeletons/, in the coordinates of the original image, plus generated/manifest.csv
(one row per glyph, confidence-ranked) and generated/wedges.csv (one row per wedge). Human
annotations in skeletons/ are never touched by `run`.

A glyph whose image is byte-identical to a human-labelled glyph in another font gets a copy
of that human skeleton instead of a prediction. Blank images (the font lacks the sign, or
rendering failed upstream) are listed and skipped.

`promote` is the only command that writes into skeletons/: it copies reviewer-accepted
skeletons there (a corrected version from --corrected wins over the generated one), refuses
to overwrite, and adds new codepoints to the training split so val and test stay fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from .skeleton_io import write_wedges
from .data import ALL_FONTS, KIND_WEDGE, PROMOTED_LOG, Pair, Wedge, find_pairs, load_metadata, load_skeleton
from .imageprep import Transform, prepare_image  # noqa: F401  (re-exported)
from .render import overlay_wedges, save_sheets, PALETTE

ACCEPT_WORDS = {"y", "yes", "1", "true", "accept", "accepted", "ok"}
DOUBTFUL = 0.3          # a wedge below this confidence is one an annotator should look at


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- run

def run(args) -> int:
    import torch  # noqa: F401  (fail early if the learned stack is missing)
    from .hybrid import HybridPredictor, glyph_confidence
    from .classical import _image_angle, _outside_arc
    from .learned.predictor import LearnedPredictor

    root, out, rep = Path(args.root), Path(args.out), Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    (rep / "review").mkdir(parents=True, exist_ok=True)
    meta = load_metadata(root)
    labelled = {p.key: p for p in find_pairs(root, include_promoted=True)}     # promoted glyphs are labelled too: never re-predict them
    labelled_by_hash = {}
    for p in labelled.values():
        if not p.promoted:                                                       # only original human annotations are copied to identical images
            labelled_by_hash.setdefault(_md5(p.image_path), p)

    hybrid = HybridPredictor(LearnedPredictor.from_checkpoint(args.checkpoint))
    rows, wedge_rows, cache, review_cells = [], [], {}, {}
    fonts = [f for f in ALL_FONTS if (root / "prototypes" / f).exists()]
    for font in fonts:
        for img_path in sorted((root / "prototypes" / font).glob("*/*.png")):
            cp = img_path.stem
            if f"{font}/{cp}" in labelled:
                continue
            base = {"font": font, "codepoint": cp, "name": meta.get(cp, {}).get("name", ""),
                    "in_font_per_metadata": meta.get(cp, {}).get(font, "")}
            image, tf = prepare_image(img_path)
            if not (image < 128).any():
                rows.append({**base, "source": "blank", "n_wedges": 0, "confidence": "", "rare_orientation_wedges": "", "scale": ""})
                continue
            h = _md5(img_path)
            adf, con = out / font / f"{cp}_adf.csv", out / font / f"{cp}_con.csv"
            if h in labelled_by_hash:                                   # a human already annotated this exact image
                src = labelled_by_hash[h]
                adf.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src.adf_path, adf)
                shutil.copyfile(src.con_path, con)
                n = len(load_skeleton(Pair(font, cp, img_path, adf, con)).wedges)
                rows.append({**base, "source": f"copied_human:{src.key}", "n_wedges": n, "confidence": 1.0,
                             "rare_orientation_wedges": "", "scale": 1.0})
                continue
            if h not in cache:
                cache[h] = hybrid.predict_scored(image, None)
            scored = cache[h]
            scored = sorted(scored, key=lambda s: (s.wedge.apex[0], s.wedge.apex[1]))     # strokes left to right
            with Image.open(img_path) as _im:
                orig_size = _im.size
            rare, clamped = 0, 0
            out_wedges = []
            for i, s in enumerate(scored, start=1):
                w = s.wedge
                t = np.array(w.tail) - np.array(w.apex)
                is_rare = bool(np.linalg.norm(t) >= 4 and _outside_arc(_image_angle(t), -40.0, 130.0) > 0)
                rare += is_rare
                orig = tf.to_original(np.array([w.apex, w.head_a, w.head_b, w.tail]))
                inside = np.clip(orig, [0, 0], [orig_size[0] - 1, orig_size[1] - 1])       # downstream tools need points in the image
                clamped += int(np.abs(inside - orig).max() > 0.5)
                out_wedges.append(Wedge(*[(float(x), float(y)) for x, y in inside]))
                wedge_rows.append({"font": font, "codepoint": cp, "stroke": i, "confidence": round(s.confidence, 4),
                                   "apex_score": round(s.apex_score, 4), "tail_source": s.tail_source, "rare_orientation": int(is_rare),
                                   "off_ink_tail": int(s.off_ink)})
            if scored:
                write_wedges(out_wedges, adf, con, font)          # layout, number format and ordering of the font's originals
            rows.append({**base, "source": "model" if scored else "model_empty", "n_wedges": len(scored),
                         "confidence": round(glyph_confidence(scored), 4), "rare_orientation_wedges": rare,
                         "off_ink_tails": sum(s.off_ink for s in scored), "clamped_wedges": clamped,
                         "doubtful_wedges": sum(s.confidence < DOUBTFUL for s in scored),
                         "mean_wedge_confidence": round(float(np.mean([s.confidence for s in scored])), 4) if scored else "",
                         "scale": round(tf.sx, 5)})
            review_cells[(font, cp)] = (image, [s.wedge for s in scored], [s.confidence for s in scored])

    # ---- manifest, ranked: least confident first within each source
    # review order: least confident glyph first; glyph confidence saturates near 0 for complex signs,
    # so ties are broken by how many wedges are doubtful (most work first)
    rank = lambda r: (0 if r["source"].startswith("model") else 1, round(float(r["confidence"]), 2) if r["confidence"] != "" else 9.0,
                      -int(r.get("doubtful_wedges") or 0))
    rows.sort(key=rank)
    fields = ["font", "codepoint", "name", "source", "n_wedges", "confidence", "doubtful_wedges", "mean_wedge_confidence",
              "rare_orientation_wedges", "off_ink_tails",
              "clamped_wedges", "scale", "in_font_per_metadata"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(out / "wedges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["font", "codepoint", "stroke", "confidence", "apex_score", "tail_source", "rare_orientation", "off_ink_tail"])
        w.writeheader()
        w.writerows(wedge_rows)

    verify = verify_outputs(root, out, rows)
    packs = review_packs(rows, review_cells, rep, seed=args.seed, size=args.pack_size)
    calib = calibration(hybrid, root, args.splits)
    write_report(rep, out, rows, verify, packs, calib, args)
    with open(rep / "blank_images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["font", "codepoint", "name", "in_font_per_metadata"])
        w.writerows([[r["font"], r["codepoint"], r["name"], r["in_font_per_metadata"]] for r in rows if r["source"] == "blank"])
    print((rep / "stats.md").read_text()[:3500])
    return 0


# --------------------------------------------------------------------------- verification

def verify_outputs(root: Path, out: Path, rows: list[dict]) -> dict:
    """Reload every written skeleton through the normal loader: every stroke must parse as a
    canonical wedge, the stroke count must match the manifest, and points must lie inside the
    original image. Also checks the model<->original transform is an exact inverse."""
    bad, checked, max_rt = [], 0, 0.0
    for r in rows:
        if r["source"] in ("blank", "model_empty"):
            continue
        img = next((root / "prototypes" / r["font"]).glob(f"*/{r['codepoint']}.png"))
        adf, con = out / r["font"] / f"{r['codepoint']}_adf.csv", out / r["font"] / f"{r['codepoint']}_con.csv"
        s = load_skeleton(Pair(r["font"], r["codepoint"], img, adf, con))
        checked += 1
        if r["source"] == "model":
            if any(st.kind != KIND_WEDGE for st in s.strokes) or len(s.wedges) != int(r["n_wedges"]):
                bad.append((r["font"], r["codepoint"], "did not reload as canonical wedges"))
                continue
            with Image.open(img) as im:
                W, H = im.size
            arr = np.array([w.as_array() for w in s.wedges])
            if (arr < -0.01).any() or (arr[..., 0] > W - 0.99).any() or (arr[..., 1] > H - 0.99).any():
                bad.append((r["font"], r["codepoint"], "points outside the original image"))
            _, tf = prepare_image(img)
            max_rt = max(max_rt, float(np.abs(tf.to_original(tf.to_model(arr.reshape(-1, 2))) - arr.reshape(-1, 2)).max()))
    return {"checked": checked, "failed": bad, "max_roundtrip_error_px": max_rt}


# --------------------------------------------------------------------------- review packs

def review_packs(rows: list[dict], cells: dict, rep: Path, seed: int, size: int) -> dict:
    rng = np.random.default_rng(seed)
    packs = {}
    groups = {"assurbanipal_santakku": [r for r in rows if r["source"] == "model" and r["font"] != "Esagil"],
              "esagil": [r for r in rows if r["source"] == "model" and r["font"] == "Esagil"]}
    for gname, grp in groups.items():
        if not grp:
            continue
        by_conf = sorted(grp, key=lambda r: (round(float(r["confidence"]), 2), -int(r.get("doubtful_wedges") or 0)))
        picks = {"lowest": by_conf[:size],
                 "random": [grp[i] for i in sorted(rng.choice(len(grp), size=min(size, len(grp)), replace=False))]}
        for kind, sel in picks.items():
            name = f"{gname}_{kind}"
            sheet_cells = []
            for r in sel:
                image, wedges, confs = cells[(r["font"], r["codepoint"])]
                cols = [PALETTE[i % len(PALETTE)] for i in range(len(wedges))]
                cap = f"{r['font']} {r['codepoint']} {r['name'][:16]}\nconf {float(r['confidence']):.2f}  {r['n_wedges']} wedges, {r['doubtful_wedges']} doubtful" + \
                      (f"  rare x{r['rare_orientation_wedges']}" if r["rare_orientation_wedges"] else "") + \
                      (f"  OFF-INK x{r['off_ink_tails']}" if r.get("off_ink_tails") else "")
                sheet_cells.append((overlay_wedges(image, wedges, cols, scale=2), cap))
            pages = save_sheets(sheet_cells, rep / "review", name, per_page=12, cols=4, title=f"Review pack: {name}")
            csv_path = rep / "review" / f"review_{name}.csv"
            kept = {}
            if csv_path.exists():           # a rerun must never wipe decisions a reviewer already recorded
                with open(csv_path, newline="") as f:
                    kept = {(x["font"], x["codepoint"]): (x.get("accept", ""), x.get("notes", "")) for x in csv.DictReader(f)}
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["font", "codepoint", "name", "confidence", "n_wedges", "doubtful_wedges", "accept", "notes"])
                w.writerows([[r["font"], r["codepoint"], r["name"], r["confidence"], r["n_wedges"], r["doubtful_wedges"],
                              *kept.get((r["font"], r["codepoint"]), ("", ""))] for r in sel])
            packs[name] = {"csv": str(csv_path), "pages": [str(p) for p in pages], "glyphs": len(sel),
                           "mean_confidence": float(np.mean([float(r["confidence"]) for r in sel]))}
    return packs


# --------------------------------------------------------------------------- calibration

def calibration(hybrid, root: Path, splits_path: str) -> dict:
    """How often a glyph is fully correct (every wedge matched, all points within 5 px) per
    confidence band, measured on the labelled val + test glyphs the model never trained on."""
    from .hybrid import glyph_confidence
    from .data import load_image
    from .metrics import match_wedges
    from .split import load_splits, pairs_in_split
    pairs = find_pairs(root)
    splits = load_splits(splits_path)
    conf, ok = [], []
    for p in pairs_in_split(pairs, splits, "val") + pairs_in_split(pairs, splits, "test"):
        sw = hybrid.predict_scored(load_image(p), p)
        m, up, ug = match_wedges([s.wedge for s in sw], load_skeleton(p).wedges)
        good = {x.pred_index for x in m if max(x.point_errors) <= 5.0}
        conf.append(glyph_confidence(sw))
        ok.append(not up and not ug and len(good) == len(sw))
    conf, ok = np.array(conf), np.array(ok)
    bands = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.01)]
    return {"glyphs": int(len(ok)), "bands": [{"lo": lo, "hi": min(hi, 1.0), "glyphs": int(((conf >= lo) & (conf < hi)).sum()),
                                                 "fully_correct_rate": float(ok[(conf >= lo) & (conf < hi)].mean()) if ((conf >= lo) & (conf < hi)).any() else None}
                                                for lo, hi in bands]}


# --------------------------------------------------------------------------- report

def write_report(rep: Path, out: Path, rows, verify, packs, calib, args) -> None:
    from collections import Counter
    fonts = sorted({r["font"] for r in rows})
    lines = ["# Stage 5 report: batch inference and review queue", "",
             f"Model: hybrid on `{args.checkpoint}`. Output: `{out}/<Font>/<hex>_adf.csv` and `_con.csv`, same format as `skeletons/`, "
             "in original-image coordinates. `skeletons/` is not modified.", "",
             "## What was processed", "", "| Font | Unlabelled images | Blank (skipped) | Copied from an identical human-labelled image | Predicted by the model | Model found no wedges |",
             "|---|---|---|---|---|---|"]
    for f in fonts:
        c = Counter("copied" if r["source"].startswith("copied") else r["source"] for r in rows if r["font"] == f)
        lines.append(f"| {f} | {sum(c.values())} | {c['blank']} | {c['copied']} | {c['model']} | {c['model_empty']} |")
    lines += ["", "## Verification", "",
              f"{verify['checked']} written skeletons were reloaded through the normal loader. Failures: {len(verify['failed'])}. "
              f"Largest model-to-original-to-model round-trip error: {verify['max_roundtrip_error_px']:.2e} px."]
    lines += [f"- {a} {b}: {c}" for a, b, c in verify["failed"][:20]]
    lines += ["", "## Confidence of model predictions", "", "![confidence](../charts/stage5_confidence.png)", "",
              "| Font | Glyphs | Median confidence | Below 0.3 | 0.3 to 0.6 | Above 0.6 | Glyphs with a rare-orientation wedge | Glyphs with an off-ink tail | Glyphs with a point clamped to the image |", "|---|---|---|---|---|---|---|---|---|"]
    for f in fonts:
        cs = np.array([float(r["confidence"]) for r in rows if r["font"] == f and r["source"] == "model"])
        if not len(cs):
            continue
        rare = sum(1 for r in rows if r["font"] == f and r["source"] == "model" and int(r["rare_orientation_wedges"] or 0) > 0)
        clip = sum(1 for r in rows if r["font"] == f and r["source"] == "model" and int(r.get("off_ink_tails") or 0) > 0)
        clam = sum(1 for r in rows if r["font"] == f and r["source"] == "model" and int(r.get("clamped_wedges") or 0) > 0)
        lines.append(f"| {f} | {len(cs)} | {np.median(cs):.2f} | {(cs < 0.3).sum()} | {((cs >= 0.3) & (cs < 0.6)).sum()} | {(cs >= 0.6).sum()} | {rare} | {clip} | {clam} |")
    lines += ["", "## How much correction work there is", "",
              f"Glyph confidence is the weakest wedge, and these glyphs are complex (many have more than 10 wedges), so most glyphs score near zero "
              f"even when only one wedge is in doubt. The practical number is how many wedges per glyph fall below {DOUBTFUL} confidence.", "",
              "| Font | Wedges predicted | Doubtful wedges | Glyphs with 0 doubtful | 1 | 2 | 3 or more |", "|---|---|---|---|---|---|---|"]
    for f in fonts:
        m = [r for r in rows if r["font"] == f and r["source"] == "model"]
        if not m:
            continue
        d = np.array([int(r["doubtful_wedges"]) for r in m])
        nw = sum(int(r["n_wedges"]) for r in m)
        lines.append(f"| {f} | {nw} | {d.sum()} ({d.sum() / max(nw, 1):.0%}) | {(d == 0).sum()} | {(d == 1).sum()} | {(d == 2).sum()} | {(d >= 3).sum()} |")
    lines += ["", f"## What the confidence means (calibration on {calib['glyphs']} labelled val + test glyphs)", "",
              "| Confidence band | Labelled glyphs | Fully correct within 5 px |", "|---|---|---|"]
    expected = 0.0
    in_domain = np.array([float(r["confidence"]) for r in rows if r["source"] == "model" and r["font"] != "Esagil"])
    for b in calib["bands"]:
        rate = b["fully_correct_rate"]
        lines.append(f"| {b['lo']:.1f} to {b['hi']:.1f} | {b['glyphs']} | {'n/a' if rate is None else f'{rate:.2f}'} |")
        if rate is not None:
            expected += rate * ((in_domain >= b["lo"]) & (in_domain < (b["hi"] if b["hi"] < 1 else 1.01))).sum()
    if len(in_domain):
        lines += ["", f"Applied to the {len(in_domain)} Assurbanipal and Santakku predictions, this suggests roughly {expected:.0f} "
                  f"({expected / len(in_domain):.0%}) need no correction at all. Esagil has no held-out human annotations, so this "
                  "calibration cannot be checked on it; its acceptance rate has to come from the review pack."]
    lines += ["", "## Review packs", "", "Fill the `accept` column (y or n) and optionally `notes`, then run "
              "`python -m protosnap.stage5 acceptance <csv>`. The random pack estimates the overall acceptance rate; "
              "the lowest-confidence pack is where correction effort pays most.", "",
              "| Pack | Glyphs | Mean confidence | Review sheet | Contact sheets |", "|---|---|---|---|---|"]
    for name, p in packs.items():
        lines.append(f"| {name} | {p['glyphs']} | {p['mean_confidence']:.2f} | `{p['csv']}` | {', '.join('`' + x + '`' for x in p['pages'])} |")
    (rep / "stats.md").write_text("\n".join(lines) + "\n")
    with open(rep / "summary.json", "w") as f:
        json.dump({"verify": {**verify, "failed": [list(x) for x in verify["failed"]]}, "packs": packs, "calibration": calib}, f, indent=1)
    try:
        from .charts import chart_confidence
        chart_confidence(rows, rep.parent / "charts")
    except Exception as e:      # charts are a convenience; never fail the run on them
        print(f"[stage5] chart skipped: {e}")


# --------------------------------------------------------------------------- acceptance / promote

def _read_review(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def acceptance(args) -> int:
    rows = [r for r in _read_review(Path(args.review)) if r.get("accept", "").strip()]
    if not rows:
        print("no rows have the accept column filled in yet")
        return 1
    acc = np.array([r["accept"].strip().lower() in ACCEPT_WORDS for r in rows])
    conf = np.array([float(r["confidence"]) for r in rows])
    print(f"reviewed {len(rows)} glyphs: accepted {acc.sum()} ({acc.mean():.1%})")
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  confidence {lo:.1f} to {min(hi, 1.0):.1f}: {acc[m].sum()}/{m.sum()} accepted ({acc[m].mean():.1%})")
    return 0


def promote(args) -> int:
    from .split import load_splits, save_splits
    root, gen = Path(args.root), Path(args.generated)
    corrected = Path(args.corrected) if args.corrected else None
    rows = [r for r in _read_review(Path(args.review)) if r.get("accept", "").strip().lower() in ACCEPT_WORDS]
    splits = load_splits(args.splits)
    known = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    done, skipped, new_cps, log = 0, [], [], []
    for r in rows:
        font, cp = r["font"], r["codepoint"]
        hand = bool(corrected and (corrected / font / f"{cp}_adf.csv").exists())
        src_dir = corrected / font if hand else gen / font
        dst = root / "skeletons" / font
        if (dst / f"{cp}_adf.csv").exists():
            skipped.append(f"{font}/{cp} already has a human skeleton")
            continue
        if not (src_dir / f"{cp}_adf.csv").exists():
            skipped.append(f"{font}/{cp} has no generated or corrected files")
            continue
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            for suffix in ("_adf.csv", "_con.csv"):
                shutil.copyfile(src_dir / f"{cp}{suffix}", dst / f"{cp}{suffix}")
        done += 1
        where = next((n for n in ("train", "val", "test") if cp in splits[n]), "train (new codepoint)")
        log.append([font, cp, "corrected_by_hand" if hand else "accepted_as_generated", where, date.today().isoformat()])
        if cp not in known:
            new_cps.append(cp)
            known.add(cp)
    if new_cps and not args.dry_run:
        splits["train"] = sorted(set(splits["train"]) | set(new_cps))
        splits.setdefault("meta", {}).setdefault("added_by_promote", []).extend(new_cps)
        save_splits(splits, args.splits)
    if log and not args.dry_run:
        logp = root / PROMOTED_LOG
        logp.parent.mkdir(parents=True, exist_ok=True)
        fresh = not logp.exists()
        with open(logp, "a", newline="") as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(["font", "codepoint", "origin", "codepoint_split", "date"])
            w.writerows(log)
    from collections import Counter
    c = Counter((x[2], x[3].split(" ")[0]) for x in log)
    print("  by origin and split:", {f"{o} / {s}": n for (o, s), n in sorted(c.items())})
    held = sum(n for (o, s), n in c.items() if s in ("val", "test"))
    if held:
        print(f"  {held} of these share a codepoint with validation or test: they are published as labels but never trained on or evaluated against")
    print(f"{'would promote' if args.dry_run else 'promoted'} {done} skeletons into skeletons/; {len(new_cps)} new codepoints "
          f"{'would be' if args.dry_run else 'were'} added to the train split; skipped {len(skipped)}")
    for s in skipped[:20]:
        print("  skipped:", s)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", default=".")
    r.add_argument("--checkpoint", default="models/shipped.pt")
    r.add_argument("--out", default="generated")
    r.add_argument("--report", default="reports/stage5")
    r.add_argument("--splits", default="data/splits.json")
    r.add_argument("--pack-size", type=int, default=48)
    r.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("acceptance")
    a.add_argument("review")
    p = sub.add_parser("promote")
    p.add_argument("review")
    p.add_argument("--root", default=".")
    p.add_argument("--generated", default="generated")
    p.add_argument("--corrected", default=None, help="directory of hand-corrected <Font>/<hex>_adf.csv files that override generated ones")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return {"run": run, "acceptance": acceptance, "promote": promote}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
