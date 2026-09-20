from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from protosnap import find_pairs, load_image, load_skeleton
from protosnap.learned.data import (AugmentConfig, GlyphDataset, affine_matrix, apply_affine_image, apply_affine_points,
                                    augment, skeleton_arrays, to_input)
from protosnap.learned.decode import DecodeConfig, decode
from protosnap.learned.model import UNet, detector_loss
from protosnap.learned.targets import build_targets
from protosnap.metrics import match_wedges

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pairs():
    return {p.key: p for p in find_pairs(ROOT)}


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def test_targets_decode_roundtrip(pairs):
    """Perfect heatmaps and offsets must decode back to the ground truth."""
    p = pairs["Santakku/0x1200f"]
    s = load_skeleton(p)
    wedges, ign = skeleton_arrays(s)
    t = build_targets(wedges, ign)
    out = {"apex": torch.from_numpy(_logit(t["apex_hm"])), "aux": torch.from_numpy(_logit(t["aux_hm"])),
           "reg": torch.from_numpy(t["reg"])}
    pred, scores = decode(out, DecodeConfig(threshold=0.5))
    m, up, ug = match_wedges(pred, s.wedges)
    assert len(m) == len(s.wedges) and not up and not ug
    assert max(x.corner_error for x in m) < 0.01


def test_ignore_mask_covers_triangle_strokes(pairs):
    s = load_skeleton(pairs["Santakku/0x12232"])        # MUSH: all strokes are triangles
    wedges, ign = skeleton_arrays(s)
    assert len(wedges) == 0 and len(ign) > 0
    t = build_targets(wedges, ign)
    assert t["valid"].min() == 0 and t["apex_hm"].max() == 0


def test_affine_moves_image_and_points_together():
    rng = np.random.default_rng(1)
    img = np.full((256, 256), 255, np.uint8)
    pts = np.array([[80.0, 90.0], [170.0, 120.0], [128.0, 160.0]])
    for x, y in pts.astype(int):
        img[y - 2:y + 3, x - 2:x + 3] = 0
    M = affine_matrix(rng, AugmentConfig(rotate_deg=8, translate=10), 256)
    out = apply_affine_image(M, img)
    moved = apply_affine_points(M, pts)
    from scipy import ndimage as ndi
    lab, n = ndi.label(out < 128)
    cents = np.array([c[::-1] for c in ndi.center_of_mass(out < 128, lab, range(1, n + 1))])
    assert n == 3
    for q in moved:
        assert np.linalg.norm(cents - q, axis=1).min() < 0.8


def test_augment_preserves_canonical_head_order(pairs):
    p = pairs["Santakku/0x12000"]
    wedges, ign = skeleton_arrays(load_skeleton(p))
    for seed in range(5):
        img, w, _ = augment(load_image(p), wedges, ign, np.random.default_rng(seed))
        assert img.shape == (256, 256) and img.dtype == np.uint8
        a, b = w[:, 1] - w[:, 0], w[:, 2] - w[:, 0]
        assert (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0).all()


def test_input_channels(pairs):
    img = load_image(pairs["Santakku/0x12000"])
    assert to_input(img, False).shape == (1, 256, 256)
    x = to_input(img, True)
    assert x.shape == (3, 256, 256) and x.max() <= 1 and x.min() >= 0


def test_model_forward_and_loss(pairs):
    p = pairs["Santakku/0x12000"]
    ds = GlyphDataset([p], {p.key: load_skeleton(p)}, train=True)
    b = ds[0]
    batch = {k: v[None] for k, v in b.items()}
    model = UNet(1, base=8, depth=4)
    out = model(batch["image"])
    assert out["apex"].shape == (1, 1, 256, 256) and out["aux"].shape == (1, 2, 256, 256) and out["reg"].shape == (1, 8, 256, 256)
    loss, parts = detector_loss(out, batch)
    assert torch.isfinite(loss) and set(parts) == {"apex", "aux", "reg", "total"}
    loss.backward()


def test_charts_render_from_saved_results(tmp_path):
    """The bar charts are drawn from saved JSON only, so they must regenerate without a GPU."""
    from protosnap.charts import make_all
    reports = ROOT / "reports"
    if not (reports / "stage2" / "json" / "classical_test.json").exists() or not list((reports / "stage3").glob("*/test.json")):
        pytest.skip("no saved results to chart")
    made = make_all(reports, tmp_path, "test")
    assert {p.name for p in made} >= {"metrics_test.png", "metrics_by_font_test.png"}
    assert all(p.stat().st_size > 20_000 for p in made)
