import csv
import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from protosnap import Pair, load_skeleton
from protosnap.editor import EditorState, serve

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "generated"
pytestmark = pytest.mark.skipif(not (GEN / "manifest.csv").exists(), reason="stage 5 has not been run")


@pytest.fixture()
def server(tmp_path):
    rows = [r for r in csv.DictReader(open(GEN / "manifest.csv")) if r["source"] == "model"][:3]
    review = tmp_path / "review.csv"
    with open(review, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["font", "codepoint", "name", "confidence", "n_wedges", "doubtful_wedges", "accept", "notes"])
        for i, r in enumerate(rows):
            w.writerow([r["font"], r["codepoint"], r["name"], r["confidence"], r["n_wedges"], r["doubtful_wedges"], "n" if i == 0 else "", "tail short" if i == 0 else ""])
    state = EditorState(ROOT, GEN, tmp_path / "reviewed", review, "all")
    httpd = serve(state, 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rows, review, tmp_path / "reviewed"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read(), r.headers.get("Content-Type")


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_page_list_glyph_and_image(server):
    base, rows, review, reviewed = server
    html, ctype = _get(base + "/")
    assert b"Skeleton editor" in html and ctype.startswith("text/html")
    listing = json.loads(_get(base + "/api/glyphs")[0])
    assert len(listing["rows"]) == 3 and listing["rows"][0]["accept"] == "n" and listing["rows"][0]["corrected"] is False
    r = rows[0]
    g = json.loads(_get(f"{base}/api/glyph?font={r['font']}&cp={r['codepoint']}")[0])
    assert g["source"] == "generated" and len(g["wedges"]) == int(r["n_wedges"]) and g["width"] > 0
    assert set(g["wedges"][0]) >= {"apex", "head_a", "head_b", "tail", "confidence"}
    png, ctype = _get(f"{base}/image?font={r['font']}&cp={r['codepoint']}")
    assert ctype == "image/png" and png[:4] == b"\x89PNG"


def test_save_writes_reviewed_files_and_marks_the_row(server):
    base, rows, review, reviewed = server
    r = rows[0]
    g = json.loads(_get(f"{base}/api/glyph?font={r['font']}&cp={r['codepoint']}")[0])
    wedges = [{k: w[k] for k in ("apex", "head_a", "head_b", "tail")} for w in g["wedges"]]
    wedges[0]["tail"] = [wedges[0]["tail"][0] + 7.5, wedges[0]["tail"][1]]          # drag one tail
    wedges = wedges[:-1] if len(wedges) > 2 else wedges                               # and delete a wedge
    out = _post(base + "/api/save", {"font": r["font"], "codepoint": r["codepoint"], "wedges": wedges})
    assert out["wedges"] == len(wedges)
    adf = reviewed / r["font"] / f"{r['codepoint']}_adf.csv"
    s = load_skeleton(Pair(r["font"], r["codepoint"], ROOT, adf, reviewed / r["font"] / f"{r['codepoint']}_con.csv"))
    assert s.is_clean and len(s.wedges) == len(wedges)
    assert any(abs(w.tail[0] - wedges[0]["tail"][0]) <= 0.5 for w in s.wedges)      # layout A fonts store integers, like their originals
    from protosnap.skeleton_io import deviations
    assert deviations(adf, reviewed / r["font"] / f"{r['codepoint']}_con.csv", r["font"]) == []
    # generated files are untouched, the review row is now accepted with a note, and the glyph reloads from reviewed/
    assert (GEN / r["font"] / f"{r['codepoint']}_adf.csv").exists()
    row = next(x for x in csv.DictReader(open(review)) if x["codepoint"] == r["codepoint"])
    assert row["accept"] == "y" and "corrected in editor" in row["notes"] and "tail short" in row["notes"]
    g2 = json.loads(_get(f"{base}/api/glyph?font={r['font']}&cp={r['codepoint']}")[0])
    assert g2["source"] == "reviewed" and len(g2["wedges"]) == len(wedges)
    _post(base + "/api/revert", {"font": r["font"], "codepoint": r["codepoint"]})
    assert not adf.exists()


def test_review_endpoint_and_input_validation(server):
    base, rows, review, reviewed = server
    r = rows[1]
    _post(base + "/api/review", {"font": r["font"], "codepoint": r["codepoint"], "accept": "n", "notes": "two tails swapped"})
    row = next(x for x in csv.DictReader(open(review)) if x["codepoint"] == r["codepoint"])
    assert row["accept"] == "n" and row["notes"] == "two tails swapped"
    for bad in ("/api/glyph?font=../../etc&cp=0x12000", "/api/glyph?font=Esagil&cp=../passwd", "/image?font=Esagil&cp=0x12000/../x"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + bad)
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError):
        _post(base + "/api/save", {"font": r["font"], "codepoint": r["codepoint"], "wedges": [{"apex": [1, 2]}]})


# ----------------------------------------------------------------------------- master review file

def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_master_review_covers_everything_and_keeps_decisions(tmp_path):
    from protosnap.editor import MASTER_NAME, build_master_review
    gen, rev = tmp_path / "generated", tmp_path / "review"
    gen.mkdir()
    rev.mkdir()
    _write(gen / "manifest.csv", ["font", "codepoint", "name", "source", "n_wedges", "confidence", "doubtful_wedges"],
           [["Esagil", "0x12000", "A", "model", 3, 0.5, 0], ["Esagil", "0x12001", "B", "model", 9, 0.01, 4],
            ["Santakku", "0x12002", "C", "copied_human:Assurbanipal/0x12002", 5, 1.0, ""], ["Santakku", "0x12003", "D", "blank", 0, "", ""]])
    hdr = ["font", "codepoint", "name", "confidence", "n_wedges", "doubtful_wedges", "accept", "notes"]
    _write(rev / "review_esagil_lowest.csv", hdr, [["Esagil", "0x12001", "B", 0.01, 9, 4, "n", "tail cut short"]])
    master, info = build_master_review(gen, rev)
    rows = {r["codepoint"]: r for r in csv.DictReader(open(master))}
    assert master.name == MASTER_NAME and set(rows) == {"0x12000", "0x12001", "0x12002"}        # blank images are not listed
    assert rows["0x12001"]["accept"] == "n" and rows["0x12001"]["notes"] == "tail cut short" and info["carried_from_packs"] == 1
    assert rows["0x12002"]["source"] == "copied_human" and rows["0x12000"]["accept"] == ""
    # a decision made in the master survives a rebuild and is not overwritten by a pack
    rows["0x12001"]["accept"], rows["0x12000"]["accept"] = "y", "y"
    _write(master, list(rows["0x12000"].keys()), [list(r.values()) for r in rows.values()])
    master, info = build_master_review(gen, rev)
    again = {r["codepoint"]: r for r in csv.DictReader(open(master))}
    assert again["0x12001"]["accept"] == "y" and again["0x12000"]["accept"] == "y" and info["carried_from_packs"] == 0


def test_master_file_works_with_acceptance_and_font_filter(tmp_path):
    from types import SimpleNamespace
    from protosnap.editor import build_master_review
    from protosnap.stage5 import acceptance
    master, info = build_master_review(GEN, tmp_path)
    assert info["rows"] > 900
    state = EditorState(ROOT, GEN, tmp_path / "reviewed", master, "all", font="Esagil")
    rows = state.rows()
    assert rows and all(r["font"] == "Esagil" for r in rows) and "doubtful_wedges" in rows[0]
    assert acceptance(SimpleNamespace(review=str(master))) in (0, 1)      # 1 only when nothing is decided yet
