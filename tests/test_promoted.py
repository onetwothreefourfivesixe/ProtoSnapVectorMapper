"""Guards around labels promoted from reviewed model output."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from protosnap import find_pairs, load_skeleton
from protosnap.data import PROMOTED_LOG
from protosnap.split import load_splits, pairs_in_split, training_pairs

ROOT = Path(__file__).resolve().parents[1]
has_promoted = (ROOT / PROMOTED_LOG).exists()


def test_default_view_is_the_original_human_corpus():
    pairs = find_pairs(ROOT)
    assert len(pairs) == 449 and not any(p.promoted for p in pairs)
    assert {p.font for p in pairs} == {"Assurbanipal", "Santakku"}


def test_promoted_lookup_with_a_synthetic_log(tmp_path):
    """find_pairs marks exactly the logged skeletons and hides them by default."""
    import shutil
    root = tmp_path
    src = next(p for p in find_pairs(ROOT) if p.font == "Santakku")
    (root / "prototypes" / "Santakku" / "x").mkdir(parents=True)
    (root / "skeletons" / "Santakku").mkdir(parents=True)
    (root / "data").mkdir()
    shutil.copy(src.image_path, root / "prototypes" / "Santakku" / "x" / src.image_path.name)
    shutil.copy(src.adf_path, root / "skeletons" / "Santakku" / src.adf_path.name)
    shutil.copy(src.con_path, root / "skeletons" / "Santakku" / src.con_path.name)
    assert [p.promoted for p in find_pairs(root)] == [False]
    with open(root / PROMOTED_LOG, "w", newline="") as f:
        csv.writer(f).writerows([["font", "codepoint", "origin", "codepoint_split", "date"], ["Santakku", src.codepoint, "accepted_as_generated", "train", "2026-01-01"]])
    assert find_pairs(root) == []
    got = find_pairs(root, include_promoted=True)
    assert len(got) == 1 and got[0].promoted


@pytest.mark.skipif(not has_promoted, reason="nothing has been promoted")
def test_promoted_labels_never_reach_evaluation_and_never_leak_into_training():
    splits = load_splits(ROOT / "data" / "splits.json")
    every = find_pairs(ROOT, include_promoted=True)
    promoted = [p for p in every if p.promoted]
    assert promoted and all(load_skeleton(p).is_clean for p in promoted)
    held_out = set(splits["val"]) | set(splits["test"])
    train = training_pairs(every, splits)
    assert not any(p.codepoint in held_out for p in train)                       # no held-out sign in training, in any font
    assert any(p.promoted for p in train)
    for name in ("val", "test"):
        ev = pairs_in_split(find_pairs(ROOT), splits, name)
        assert len(ev) == 45 and not any(p.promoted for p in ev)                 # evaluation sets are frozen
    before = ROOT / "data" / "splits.before_promote.json"
    if before.exists():
        b = json.load(open(before))
        assert b["val"] == splits["val"] and b["test"] == splits["test"] and set(b["train"]) <= set(splits["train"])


@pytest.mark.skipif(not has_promoted, reason="nothing has been promoted")
def test_dataset_rescales_esagil_labels_into_model_space():
    torch = pytest.importorskip("torch")
    from protosnap.learned.data import GlyphDataset
    esagil = [p for p in find_pairs(ROOT, include_promoted=True) if p.promoted and p.font == "Esagil"][:3]
    if not esagil:
        pytest.skip("no promoted Esagil glyphs")
    sk = {p.key: load_skeleton(p) for p in esagil}
    ds = GlyphDataset(esagil, sk, train=False, extra_channels=True)
    for i, p in enumerate(esagil):
        item = ds[i]
        assert item["image"].shape == (3, 256, 256) and item["apex_hm"].shape == (1, 256, 256)
        n = len(sk[p.key].wedges)
        assert int((item["apex_hm"] > 0.99).sum()) >= max(1, n - 1)              # apexes landed inside the canvas
        ys, xs = np.nonzero(item["apex_hm"][0].numpy() > 0.99)
        ink = item["image"][0].numpy() > 0.5
        near = [ink[max(0, y - 4):y + 5, max(0, x - 4):x + 5].any() for y, x in zip(ys, xs)]
        assert np.mean(near) > 0.9                                                  # and they sit on the rescaled ink
