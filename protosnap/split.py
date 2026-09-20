"""Codepoint-level train/val/test split, stratified by which fonts contain the codepoint.

The split unit is the codepoint, not the file: a sign that exists in both fonts is
assigned to one split for both fonts, so the same sign never appears in train and test.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .data import Pair

SPLIT_NAMES = ("train", "val", "test")

# Assurbanipal renders HI (0x1212d) and SHAR2 (0x122b9) as pixel-identical images with
# different skeletons. One of them is pinned to train so the ambiguity never counts as an
# evaluation error in either direction.
DEFAULT_FORCE_TRAIN: tuple[str, ...] = ("0x122b9",)


def make_splits(
    pairs: Iterable[Pair],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    force_train: Iterable[str] = DEFAULT_FORCE_TRAIN,
) -> dict:
    """Return {"meta": {...}, "train": [codepoints], "val": [...], "test": [...]}."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    fonts_of: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        fonts_of[p.codepoint].add(p.font)
    force = set(force_train)

    groups: dict[str, list[str]] = defaultdict(list)
    for cp, fonts in fonts_of.items():
        groups["+".join(sorted(fonts))].append(cp)

    rng = random.Random(seed)
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    group_counts: dict[str, dict[str, int]] = {}
    for gname in sorted(groups):
        cps = sorted(groups[gname])
        pinned = [c for c in cps if c in force]
        free = [c for c in cps if c not in force]
        rng.shuffle(free)
        n = len(cps)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        val, test, train = free[:n_val], free[n_val:n_val + n_test], free[n_val + n_test:] + pinned
        out["train"] += train
        out["val"] += val
        out["test"] += test
        group_counts[gname] = {"train": len(train), "val": len(val), "test": len(test)}

    for name in SPLIT_NAMES:
        out[name].sort()
    return {
        "meta": {
            "unit": "codepoint",
            "seed": seed,
            "fractions": list(fractions),
            "force_train": sorted(force & set(fonts_of)),
            "groups": group_counts,
        },
        **out,
    }


def save_splits(splits: dict, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)


def load_splits(path: Path | str) -> dict:
    with open(path) as f:
        return json.load(f)


def pairs_in_split(pairs: Iterable[Pair], splits: dict, name: str) -> list[Pair]:
    cps = set(splits[name])
    return [p for p in pairs if p.codepoint in cps]


def training_pairs(all_pairs: Iterable[Pair], splits: dict) -> list[Pair]:
    """Training set when promoted labels are in play: every pair whose codepoint is in the train
    split, original or promoted. Promoted pairs whose codepoint belongs to val or test are left
    out (the split is by codepoint, so training on them would leak a held-out sign), and
    find_pairs never returns promoted pairs to the evaluation code in the first place."""
    cps = set(splits["train"])
    return [p for p in all_pairs if p.codepoint in cps]
