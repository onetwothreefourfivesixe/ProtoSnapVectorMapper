"""Stage 0 runner: index pairs, canonicalize skeletons, build splits, render contact sheets,
and write a stats report for review.

    python -m protosnap.stage0 [--root .] [--out reports/stage0] [--splits data/splits.json] [--seed 0]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .data import KIND_WEDGE, FONTS_WITH_SKELETONS, find_pairs, load_image, load_metadata, load_skeleton
from .render import overlay, save_sheets
from .split import SPLIT_NAMES, make_splits, save_splits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/stage0")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true", help="skip contact sheet rendering")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(root)
    pairs = find_pairs(root)
    skels = {p.key: load_skeleton(p) for p in pairs}

    # ---- stats
    kinds = Counter()
    kinds_by_font = defaultdict(Counter)
    orphan_strokes = 0
    strokes_per_glyph = Counter()
    flagged = []
    for p in pairs:
        s = skels[p.key]
        strokes_per_glyph[len(s.strokes)] += 1
        for st in s.strokes:
            kinds[st.kind] += 1
            kinds_by_font[p.font][st.kind] += 1
            if st.kind == KIND_WEDGE and st.orphans:
                orphan_strokes += 1
        if not s.is_clean:
            flagged.append((p, s))

    # ---- splits
    splits = make_splits(pairs, seed=args.seed)
    save_splits(splits, args.splits)
    split_of = {cp: name for name in SPLIT_NAMES for cp in splits[name]}
    split_by_font = defaultdict(Counter)
    for p in pairs:
        split_by_font[p.font][split_of[p.codepoint]] += 1

    # ---- report
    lines = ["# Stage 0 report", ""]
    lines += ["## Pairs", "", "| Font | Pairs | Clean | Flagged |", "|---|---|---|---|"]
    for font in FONTS_WITH_SKELETONS:
        fp = [p for p in pairs if p.font == font]
        nflag = sum(1 for p, _ in flagged if p.font == font)
        lines.append(f"| {font} | {len(fp)} | {len(fp) - nflag} | {nflag} |")
    lines.append(f"| **Total** | {len(pairs)} | {len(pairs) - len(flagged)} | {len(flagged)} |")

    lines += ["", "## Stroke kinds", "", "| Font | wedge | of which with orphan points | triangle | malformed |", "|---|---|---|---|---|"]
    for font in FONTS_WITH_SKELETONS:
        k = kinds_by_font[font]
        orph = sum(1 for p in pairs if p.font == font for st in skels[p.key].strokes if st.kind == KIND_WEDGE and st.orphans)
        lines.append(f"| {font} | {k['wedge']} | {orph} | {k['triangle']} | {k['malformed']} |")
    lines.append(f"| **Total** | {kinds['wedge']} | {orphan_strokes} | {kinds['triangle']} | {kinds['malformed']} |")

    lines += ["", "## Strokes per glyph", "", "| Strokes | Glyphs |", "|---|---|"]
    lines += [f"| {n} | {c} |" for n, c in sorted(strokes_per_glyph.items())]

    lines += ["", f"## Split (unit = codepoint, seed = {args.seed}, fractions = {splits['meta']['fractions']})", "",
              "| Font | train | val | test |", "|---|---|---|---|"]
    for font in FONTS_WITH_SKELETONS:
        c = split_by_font[font]
        lines.append(f"| {font} | {c['train']} | {c['val']} | {c['test']} |")
    lines += ["", "Codepoints per font-group:", "", "| Group | train | val | test |", "|---|---|---|---|"]
    for g, c in splits["meta"]["groups"].items():
        lines.append(f"| {g} | {c['train']} | {c['val']} | {c['test']} |")
    lines.append(f"\nPinned to train: {', '.join(splits['meta']['force_train']) or 'none'}")

    corrected = [p for p in pairs if p.corrected]
    lines += ["", "## Corrections applied", "", f"{len(corrected)} glyph(s) read from `data/corrections/` instead of `skeletons/`:", ""]
    lines += [f"- {p.font} {p.codepoint} {meta.get(p.codepoint, {}).get('name', '')}" for p in corrected] or ["- none"]

    lines += ["", "## Flagged glyphs", "", "| Font | Codepoint | Name | Split | Flags |", "|---|---|---|---|---|"]
    for p, s in flagged:
        name = meta.get(p.codepoint, {}).get("name", "")
        lines.append(f"| {p.font} | {p.codepoint} | {name} | {split_of[p.codepoint]} | {'; '.join(s.flags)} |")

    (out / "stats.md").write_text("\n".join(lines) + "\n")

    # ---- contact sheets
    if not args.no_sheets:
        sheet_dir = out / "contact_sheets"
        for font in FONTS_WITH_SKELETONS:
            cells = []
            for p in [p for p in pairs if p.font == font]:
                s = skels[p.key]
                name = meta.get(p.codepoint, {}).get("name", "") or "?"
                cap = f"{p.codepoint}  {name[:22]}\n{len(s.strokes)} strokes  {split_of[p.codepoint]}"
                if s.flags:
                    cap += "  FLAG"
                if p.corrected:
                    cap += "  CORR"
                cells.append((overlay(load_image(p), s, scale=1), cap))
            save_sheets(cells, sheet_dir, font, title=f"{font}: glyph + skeleton overlay")
        if flagged:
            cells = []
            for p, s in flagged:
                name = meta.get(p.codepoint, {}).get("name", "") or "?"
                cap = f"{p.font} {p.codepoint} {name[:16]}\n{'; '.join(s.flags)[:40]}"
                cells.append((overlay(load_image(p), s, scale=2), cap))
            save_sheets(cells, sheet_dir, "flagged", per_page=12, cols=4, title="Flagged strokes (2x)")

    print(f"pairs={len(pairs)} wedges={kinds['wedge']} triangles={kinds['triangle']} "
          f"malformed={kinds['malformed']} flagged_glyphs={len(flagged)}")
    print(f"splits -> {args.splits}; report -> {out / 'stats.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
