import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from protosnap import Pair, find_pairs, load_skeleton
from protosnap.stage5 import Transform, acceptance, prepare_image, promote

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"


def test_transform_is_an_exact_inverse():
    tf = Transform(0.15625, 0.15625, 14.0, 88.0)
    pts = np.array([[0.0, 0.0], [331.0, 511.0], [100.5, 250.25]])
    assert np.allclose(tf.to_original(tf.to_model(pts)), pts, atol=1e-9)
    assert np.allclose(Transform().to_model(pts), pts)


def test_prepare_image_native_and_esagil():
    native = next((ROOT / "prototypes" / "Santakku").glob("*/0x12000.png"))
    img, tf = prepare_image(native)
    assert img.shape == (256, 256) and tf == Transform()
    esagil = next((ROOT / "prototypes" / "Esagil").glob("*/0x12000.png"))
    img, tf = prepare_image(esagil)
    assert img.shape == (256, 256) and img.dtype == np.uint8
    assert abs(tf.sy - 80 / 512) < 1e-3 and tf.oy == 88
    # rescaled Esagil sign A carries about as much ink as the training rendering of the same sign
    ink_e = (img < 128).sum()
    ink_s = (prepare_image(native)[0] < 128).sum()
    assert 0.7 < ink_e / ink_s < 1.4
    # ink stays inside the pasted area
    ys, xs = np.where(img < 128)
    assert ys.min() >= tf.oy and ys.max() <= tf.oy + 80


def test_tail_coverage_flags_lines_that_leave_the_ink():
    from scipy import ndimage as ndi
    from protosnap.hybrid import tail_coverage
    ink = np.zeros((64, 64), bool)
    ink[30:33, 5:40] = True                       # a horizontal stroke
    dist = ndi.distance_transform_edt(~ink)
    cov, last = tail_coverage(dist, (6, 31), (38, 31), 1.5)
    assert cov == 1.0 and tuple(last) == (38.0, 31.0)
    cov, last = tail_coverage(dist, (6, 31), (60, 31), 1.5)     # runs 20 px past the ink
    assert cov < 0.8 and 38 <= last[0] <= 42


def _review_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["font", "codepoint", "name", "confidence", "n_wedges", "doubtful_wedges", "accept", "notes"])
        w.writerows(rows)


def test_acceptance_and_promote(tmp_path, capsys):
    review = tmp_path / "review.csv"
    _review_csv(review, [["Assurbanipal", "0xAAAA1", "X", 0.7, 3, 0, "y", ""], ["Assurbanipal", "0xAAAA2", "Y", 0.1, 5, 2, "n", "tail wrong"],
                         ["Assurbanipal", "0xAAAA3", "Z", 0.4, 4, 1, "", ""]])
    assert acceptance(SimpleNamespace(review=str(review))) == 0
    assert "accepted 1 (50.0%)" in capsys.readouterr().out          # the unfilled row is not counted

    root, gen, corrected = tmp_path / "root", tmp_path / "gen", tmp_path / "fixed"
    (gen / "Assurbanipal").mkdir(parents=True)
    (root / "skeletons" / "Assurbanipal").mkdir(parents=True)
    for suffix, body in (("_adf.csv", "label,x,y\n"), ("_con.csv", "label,i,j\n")):
        (gen / "Assurbanipal" / f"0xAAAA1{suffix}").write_text(body + "generated")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"meta": {}, "train": ["0x12000"], "val": ["0x12001"], "test": ["0x12002"]}))
    ns = SimpleNamespace(review=str(review), root=str(root), generated=str(gen), corrected=None, splits=str(splits), dry_run=True)
    assert promote(ns) == 0
    assert not (root / "skeletons" / "Assurbanipal" / "0xAAAA1_adf.csv").exists()            # dry run writes nothing
    assert json.loads(splits.read_text())["train"] == ["0x12000"]

    (corrected / "Assurbanipal").mkdir(parents=True)
    for suffix in ("_adf.csv", "_con.csv"):
        (corrected / "Assurbanipal" / f"0xAAAA1{suffix}").write_text("corrected by a human")
    ns.dry_run, ns.corrected = False, str(corrected)
    assert promote(ns) == 0
    dst = root / "skeletons" / "Assurbanipal" / "0xAAAA1_adf.csv"
    assert dst.read_text() == "corrected by a human"                                            # corrected beats generated
    s = json.loads(splits.read_text())
    assert "0xAAAA1" in s["train"] and s["val"] == ["0x12001"] and s["test"] == ["0x12002"]     # val and test stay fixed
    dst.write_text("do not overwrite me")
    assert promote(ns) == 0 and dst.read_text() == "do not overwrite me"                        # never overwrites


@pytest.mark.skipif(not (GEN / "manifest.csv").exists(), reason="stage 5 has not been run")
def test_generated_outputs_are_consistent():
    rows = list(csv.DictReader(open(GEN / "manifest.csv")))
    labelled = {p.key for p in find_pairs(ROOT)}
    assert not any(f"{r['font']}/{r['codepoint']}" in labelled for r in rows)                  # never predicts over a human label
    assert {r["source"].split(":")[0] for r in rows} <= {"model", "model_empty", "copied_human", "blank"}
    model = [r for r in rows if r["source"] == "model"]
    assert len(model) > 500 and all(0.0 <= float(r["confidence"]) <= 1.0 for r in model)
    for r in model[:40] + [r for r in model if r["font"] == "Esagil"][:20]:
        img = next((ROOT / "prototypes" / r["font"]).glob(f"*/{r['codepoint']}.png"))
        s = load_skeleton(Pair(r["font"], r["codepoint"], img, GEN / r["font"] / f"{r['codepoint']}_adf.csv", GEN / r["font"] / f"{r['codepoint']}_con.csv"))
        assert s.is_clean and len(s.wedges) == int(r["n_wedges"])
    copied = [r for r in rows if r["source"].startswith("copied_human")]
    for r in copied:
        src_font, src_cp = r["source"].split(":")[1].split("/")
        assert (GEN / r["font"] / f"{r['codepoint']}_adf.csv").read_text() == (ROOT / "skeletons" / src_font / f"{src_cp}_adf.csv").read_text()
