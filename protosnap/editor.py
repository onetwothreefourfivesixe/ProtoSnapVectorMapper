"""Local skeleton editor: drag wedge points on the glyph, save corrected CSVs.

    python -m protosnap.editor                      # every generated skeleton, tracked in review_all.csv
    python -m protosnap.editor --only rejected      # just the rows marked n
    python -m protosnap.editor --font Esagil --only unreviewed
    python -m protosnap.editor --review reports/stage5/review/review_esagil_lowest.csv   # one pack only

With no --review the editor uses a master review file, reports/stage5/review/review_all.csv,
covering every skeleton under generated/ (model predictions and the copies of human labels).
It is created on first start and kept in step with generated/manifest.csv; decisions already
recorded in the pack CSVs are carried into it and never overwritten.

Open http://localhost:8765 in a browser. Saving writes data/reviewed/<Font>/<hex>_adf.csv and
_con.csv in the same format as skeletons/, in original-image coordinates, and (with --review)
marks the row accepted with a "corrected in editor" note. Nothing under skeletons/ or generated/
is modified; `python -m protosnap.stage5 promote <csv> --corrected data/reviewed` does that step.
The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .skeleton_io import write_wedges
from .data import ALL_FONTS, Pair, Wedge, load_metadata, load_skeleton

CP_RE = re.compile(r"^0x[0-9a-fA-F]{4,6}$")
REVIEW_FIELDS = ["font", "codepoint", "name", "confidence", "n_wedges", "doubtful_wedges", "accept", "notes"]
MASTER_FIELDS = REVIEW_FIELDS + ["source"]
MASTER_NAME = "review_all.csv"


def read_decisions(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(font, codepoint) -> (accept, notes) for rows where either is filled in."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {(r["font"], r["codepoint"]): (r.get("accept", "").strip(), r.get("notes", "").strip())
                for r in csv.DictReader(f) if r.get("accept", "").strip() or r.get("notes", "").strip()}


def build_master_review(generated: Path, review_dir: Path) -> tuple[Path, dict]:
    """Create or refresh the master review file so it has one row per skeleton in generated/.
    Existing decisions in the master are kept; rows still blank pick up decisions from the pack
    CSVs in the same folder. Returns the path and a small summary."""
    master = review_dir / MASTER_NAME
    review_dir.mkdir(parents=True, exist_ok=True)
    mine = read_decisions(master)
    packs: dict[tuple[str, str], tuple[str, str]] = {}
    for pack in sorted(review_dir.glob("review_*.csv")):
        if pack.name != MASTER_NAME:
            for k, v in read_decisions(pack).items():
                packs.setdefault(k, v)
    with open(generated / "manifest.csv", newline="") as f:
        manifest = [r for r in csv.DictReader(f) if r["source"] == "model" or r["source"].startswith("copied_human")]
    rows, carried = [], 0
    for r in manifest:
        key = (r["font"], r["codepoint"])
        accept, notes = mine.get(key, ("", ""))
        if not accept and not notes and key in packs:
            accept, notes = packs[key]
            carried += 1
        rows.append({"font": r["font"], "codepoint": r["codepoint"], "name": r["name"], "confidence": r["confidence"],
                     "n_wedges": r["n_wedges"], "doubtful_wedges": r.get("doubtful_wedges", ""), "accept": accept,
                     "notes": notes, "source": "copied_human" if r["source"].startswith("copied") else "model"})
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return master, {"rows": len(rows), "carried_from_packs": carried, "decided": sum(1 for r in rows if r["accept"])}


class EditorState:
    def __init__(self, root: Path, generated: Path, reviewed: Path, review_csv: Path | None, only: str, font: str | None = None):
        self.root, self.generated, self.reviewed, self.review_csv, self.only, self.font = root, generated, reviewed, review_csv, only, font
        self.lock = threading.Lock()
        self.meta = load_metadata(root)
        self.wedge_conf: dict[tuple[str, str], list[dict]] = {}
        wc = generated / "wedges.csv"
        if wc.exists():
            with open(wc, newline="") as f:
                for r in csv.DictReader(f):
                    self.wedge_conf.setdefault((r["font"], r["codepoint"]), []).append(r)

    # ---- glyph list
    def rows(self) -> list[dict]:
        if self.review_csv is not None:
            with open(self.review_csv, newline="") as f:
                rows = list(csv.DictReader(f))
        else:
            with open(self.generated / "manifest.csv", newline="") as f:
                rows = [dict(r, accept="", notes="") for r in csv.DictReader(f) if r["source"] == "model"]
        if self.only == "rejected":
            rows = [r for r in rows if r.get("accept", "").strip().lower() in ("n", "no", "0", "false", "reject", "rejected")]
        elif self.only == "unreviewed":
            rows = [r for r in rows if not r.get("accept", "").strip()]
        if self.font:
            rows = [r for r in rows if r["font"] == self.font]
        for r in rows:
            r.setdefault("source", "model")
            r["corrected"] = (self.reviewed / r["font"] / f"{r['codepoint']}_adf.csv").exists()
        return rows

    def image_path(self, font: str, cp: str) -> Path:
        return next((self.root / "prototypes" / font).glob(f"*/{cp}.png"))

    def glyph(self, font: str, cp: str) -> dict:
        from PIL import Image
        img = self.image_path(font, cp)
        with Image.open(img) as im:
            w, h = im.size
        source, skel_dir = "generated", self.generated / font
        if (self.reviewed / font / f"{cp}_adf.csv").exists():
            source, skel_dir = "reviewed", self.reviewed / font
        wedges = []
        if (skel_dir / f"{cp}_adf.csv").exists():
            s = load_skeleton(Pair(font, cp, img, skel_dir / f"{cp}_adf.csv", skel_dir / f"{cp}_con.csv"))
            conf = self.wedge_conf.get((font, cp), []) if source == "generated" else []
            for i, wd in enumerate(s.wedges):
                c = conf[i] if i < len(conf) else {}
                wedges.append({"apex": list(wd.apex), "head_a": list(wd.head_a), "head_b": list(wd.head_b), "tail": list(wd.tail),
                               "confidence": float(c["confidence"]) if c else None,
                               "off_ink": c.get("off_ink_tail") == "1" if c else False,
                               "rare": c.get("rare_orientation") == "1" if c else False})
        return {"font": font, "codepoint": cp, "name": self.meta.get(cp, {}).get("name", ""), "width": w, "height": h,
                "source": source, "wedges": wedges}

    # ---- writes
    def save(self, font: str, cp: str, wedges: list[dict]) -> dict:
        out = []
        for w in wedges:
            quad = [w.get("apex"), w.get("head_a"), w.get("head_b"), w.get("tail")]
            if any(p is None or len(p) != 2 or not all(isinstance(v, (int, float)) for v in p) for p in quad):
                raise ValueError("every wedge needs apex, head_a, head_b and tail as [x, y]")
            out.append(Wedge(*[(float(x), float(y)) for x, y in quad]))
        with self.lock:
            adf, con = self.reviewed / font / f"{cp}_adf.csv", self.reviewed / font / f"{cp}_con.csv"
            write_wedges(out, adf, con, font)       # same layout as the font's original annotations
            s = load_skeleton(Pair(font, cp, self.image_path(font, cp), adf, con))
            if not s.is_clean or len(s.wedges) != len(wedges):
                raise ValueError("saved file did not reload as canonical wedges")
            if self.review_csv is not None:
                self._update_review(font, cp, "y", "corrected in editor", append_note=True)
        return {"saved": str(adf), "wedges": len(wedges)}

    def revert(self, font: str, cp: str) -> dict:
        with self.lock:
            for suffix in ("_adf.csv", "_con.csv"):
                p = self.reviewed / font / f"{cp}{suffix}"
                if p.exists():
                    p.unlink()
        return {"reverted": True}

    def review(self, font: str, cp: str, accept: str, notes: str | None) -> dict:
        if self.review_csv is None:
            raise ValueError("start the editor with --review to record accept/reject")
        with self.lock:
            self._update_review(font, cp, accept, notes, append_note=False)
        return {"ok": True}

    def _update_review(self, font: str, cp: str, accept: str, notes: str | None, append_note: bool) -> None:
        with open(self.review_csv, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or REVIEW_FIELDS
            rows = list(reader)
        for r in rows:
            if r["font"] == font and r["codepoint"] == cp:
                r["accept"] = accept
                if notes is not None:
                    old = r.get("notes", "").strip()
                    r["notes"] = (f"{old}; {notes}" if old and notes not in old else (old or notes)) if append_note else notes
        with open(self.review_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def make_handler(state: EditorState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):   # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _key(self, q: dict) -> tuple[str, str]:
            font, cp = q.get("font", [""])[0], q.get("cp", [""])[0]
            if font not in ALL_FONTS or not CP_RE.match(cp):
                raise ValueError("bad font or codepoint")
            return font, cp

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if u.path == "/api/glyphs":
                    return self._json({"rows": state.rows(), "review": str(state.review_csv) if state.review_csv else None})
                if u.path == "/api/glyph":
                    return self._json(state.glyph(*self._key(q)))
                if u.path == "/image":
                    return self._send(200, state.image_path(*self._key(q)).read_bytes(), "image/png")
                self._json({"error": "not found"}, 404)
            except (ValueError, StopIteration, FileNotFoundError) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                font, cp = self._key({"font": [body.get("font", "")], "cp": [body.get("codepoint", "")]})
                if u.path == "/api/save":
                    return self._json(state.save(font, cp, body["wedges"]))
                if u.path == "/api/revert":
                    return self._json(state.revert(font, cp))
                if u.path == "/api/review":
                    return self._json(state.review(font, cp, body.get("accept", ""), body.get("notes")))
                self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, StopIteration, json.JSONDecodeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve(state: EditorState, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--generated", default="generated")
    ap.add_argument("--reviewed", default="data/reviewed")
    ap.add_argument("--review", default=None, help="a single review CSV to work through (default: the master file covering every generated skeleton)")
    ap.add_argument("--review-dir", default="reports/stage5/review", help="where the master review file and the pack CSVs live")
    ap.add_argument("--only", default="all", choices=["all", "rejected", "unreviewed"])
    ap.add_argument("--font", default=None, choices=list(ALL_FONTS))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.review:
        review = Path(args.review)
    else:
        review, info = build_master_review(Path(args.generated), Path(args.review_dir))
        print(f"master review file {review}: {info['rows']} skeletons, {info['decided']} already decided "
              f"({info['carried_from_packs']} carried over from the pack CSVs this start)")
    state = EditorState(Path(args.root), Path(args.generated), Path(args.reviewed), review, args.only, args.font)
    n = len(state.rows())
    httpd = serve(state, args.port)
    print(f"skeleton editor: {n} glyphs  ->  http://localhost:{args.port}   (Ctrl+C to stop)\ncorrected files go to {args.reviewed}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Skeleton editor</title>
<style>
:root{--bg:#f9f9f7;--panel:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--warn:#eb6834;--good:#0ca30c;--bad:#d03b3b}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg);display:grid;grid-template-columns:270px 1fr 290px;grid-template-rows:100%}
aside{background:var(--panel);overflow:auto;border-right:1px solid var(--line)}
aside.right{border-right:0;border-left:1px solid var(--line);padding:12px}
h1{font-size:14px;margin:12px 12px 4px}.sub{color:var(--ink2);margin:0 12px 8px;font-size:12px}
#filter{margin:0 12px 8px;width:calc(100% - 24px);padding:5px 7px;border:1px solid var(--line);border-radius:5px;font:inherit}
.ctl{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:0 12px 8px}.ctl select{font:inherit;padding:4px;border:1px solid var(--line);border-radius:5px;background:#fff;min-width:0}.ctl #fsort{grid-column:1/3}
.row{padding:6px 12px;border-top:1px solid var(--line);cursor:pointer;display:grid;grid-template-columns:1fr auto;gap:2px 8px}
.row:hover{background:#f0efec}.row.sel{background:#e4eefb}
.row .cp{font-weight:600}.row .nm{color:var(--ink2);font-size:12px;grid-column:1/3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag{font-size:11px;padding:0 6px;border-radius:9px;border:1px solid var(--line);color:var(--ink2)}
.tag.y{border-color:var(--good);color:#006300}.tag.n{border-color:var(--bad);color:var(--bad)}.tag.c{border-color:var(--accent);color:var(--accent)}
main{position:relative;overflow:hidden;background:#fff}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
#hud{position:absolute;left:10px;top:8px;background:rgba(252,252,251,.92);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:12px;color:var(--ink2);pointer-events:none}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);cursor:pointer}
button:hover{background:#f0efec}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.primary:hover{background:#256abf}
button:disabled{opacity:.45;cursor:default}
.grp{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}.grp button{flex:1 1 auto}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:14px 0 6px}
#wl{list-style:none;margin:0;padding:0;max-height:34vh;overflow:auto;border:1px solid var(--line);border-radius:6px}
#wl li{padding:4px 8px;border-top:1px solid var(--line);cursor:pointer;display:flex;justify-content:space-between;gap:8px}
#wl li:first-child{border-top:0}#wl li.sel{background:#e4eefb}#wl .w{color:var(--warn)}
textarea{width:100%;min-height:54px;border:1px solid var(--line);border-radius:6px;padding:6px;font:inherit;resize:vertical}
#msg{min-height:18px;margin-top:8px;color:var(--ink2)}#msg.err{color:var(--bad)}#msg.ok{color:#006300}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;background:#fff}
.help{color:var(--ink2);font-size:12px;line-height:1.7}
.legend span{display:inline-flex;align-items:center;gap:5px;margin-right:10px;color:var(--ink2);font-size:12px}
.sw{width:10px;height:10px;display:inline-block;border-radius:50%}
</style></head><body>
<aside><h1>Skeleton editor</h1><p class="sub" id="src"></p><input id="filter" placeholder="Filter by codepoint or name">
<div class="ctl"><select id="ffont" title="Font"></select><select id="fstat" title="Status"><option value="">Any status</option><option value="open">Open</option><option value="rejected">Rejected</option><option value="accepted">Accepted</option><option value="corrected">Corrected</option></select>
<select id="fsort" title="Order"><option value="queue">Review order</option><option value="doubt">Most doubtful wedges</option><option value="conf">Lowest confidence</option><option value="confhi">Highest confidence</option><option value="cp">Codepoint</option></select></div>
<p class="sub" id="prog"></p><div id="list"></div></aside>
<main><canvas id="cv"></canvas><div id="hud">Loading</div></main>
<aside class="right">
  <div class="grp"><button id="prev">Previous</button><button id="next">Next</button><button id="nopen">Next open</button></div>
  <div class="grp"><button id="save" class="primary">Save correction</button><button id="undo">Undo</button></div>
  <div class="grp"><button id="add">Add wedge</button><button id="del">Delete wedge</button></div>
  <div class="grp"><button id="swa">Swap tail with head A</button><button id="swb">Swap tail with head B</button></div>
  <div class="grp"><button id="revert">Discard correction</button><button id="fit">Fit view</button></div>
  <h2>Wedges</h2><ul id="wl"></ul>
  <div class="legend" style="margin-top:8px"><span><i class="sw" style="background:#2a78d6"></i>apex</span><span><i class="sw" style="background:#1baf7a;border-radius:2px"></i>head</span><span><i class="sw" style="border:2px solid #4a3aa7"></i>tail</span><span><i class="sw" style="background:#eb6834"></i>doubtful</span></div>
  <h2>Review</h2>
  <div class="grp"><button id="acc">Accept as is</button><button id="rej">Reject</button></div>
  <textarea id="notes" placeholder="Notes for this glyph"></textarea>
  <div id="msg"></div>
  <h2>Keys</h2>
  <div class="help">Drag a point to move it. Drag empty space to pan, wheel to zoom.<br><kbd>Ctrl</kbd>+<kbd>S</kbd> save &nbsp; <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo &nbsp; <kbd>Del</kbd> delete wedge<br><kbd>A</kbd> add wedge &nbsp; <kbd>F</kbd> fit &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> previous, next<br><kbd>Y</kbd> accept &nbsp; <kbd>X</kbd> reject &nbsp; <kbd>N</kbd> next open glyph<br>Hold <kbd>Shift</kbd> while dragging the apex to move the whole wedge.</div>
</aside>
<script>
"use strict";
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let rows = [], idx = -1, glyph = null, img = null, wedges = [], sel = -1, dirty = false, undoStack = [];
let view = {k: 1, ox: 0, oy: 0}, drag = null, hasReview = false;
const PTS = ["apex", "head_a", "head_b", "tail"];

function msg(t, cls) { const m = $("msg"); m.textContent = t || ""; m.className = cls || ""; }
async function api(path, body) {
  const r = await fetch(path, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : undefined);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const YES = ["y", "yes", "1", "true", "accept", "accepted", "ok"];
function statusOf(r) {
  if (r.corrected) return "corrected";
  const a = (r.accept || "").trim().toLowerCase();
  return YES.includes(a) ? "accepted" : (a ? "rejected" : "open");
}
function tagFor(r) {
  const s = statusOf(r), cls = {corrected: "c", accepted: "y", rejected: "n", open: ""}[s];
  return `<span class="tag ${cls}">${s}</span>`;
}
function visible() {
  const f = $("filter").value.trim().toLowerCase(), ff = $("ffont").value, fs = $("fstat").value, so = $("fsort").value;
  const v = [];
  rows.forEach((r, i) => {
    if (ff && r.font !== ff) return;
    if (fs && statusOf(r) !== fs) return;
    if (f && !(r.codepoint + " " + r.name + " " + r.font).toLowerCase().includes(f)) return;
    v.push(i);
  });
  const num = (r, k) => Number(r[k]) || 0;
  if (so === "doubt") v.sort((a, b) => num(rows[b], "doubtful_wedges") - num(rows[a], "doubtful_wedges") || a - b);
  else if (so === "conf") v.sort((a, b) => num(rows[a], "confidence") - num(rows[b], "confidence") || a - b);
  else if (so === "confhi") v.sort((a, b) => num(rows[b], "confidence") - num(rows[a], "confidence") || a - b);
  else if (so === "cp") v.sort((a, b) => rows[a].codepoint.localeCompare(rows[b].codepoint) || rows[a].font.localeCompare(rows[b].font));
  return v;
}
function renderList() {
  const v = visible();
  $("list").innerHTML = v.map(i => { const r = rows[i]; const d = r.doubtful_wedges !== "" && r.doubtful_wedges !== undefined ? ` &middot; ${r.doubtful_wedges} doubtful` : "";
    return `<div class="row${i === idx ? " sel" : ""}" data-i="${i}"><span class="cp">${r.font} ${r.codepoint}</span>${tagFor(r)}<span class="nm">${r.name || ""} &middot; conf ${Number(r.confidence).toFixed(2)} &middot; ${r.n_wedges} wedges${d}${r.source === "copied_human" ? " &middot; copied human label" : ""}</span></div>`; }).join("");
  const c = {open: 0, accepted: 0, rejected: 0, corrected: 0}; rows.forEach(r => c[statusOf(r)]++);
  $("prog").textContent = `${rows.length - c.open} of ${rows.length} decided \u00b7 ${c.accepted} accepted \u00b7 ${c.corrected} corrected \u00b7 ${c.rejected} rejected \u00b7 showing ${v.length}`;
}
$("list").addEventListener("click", e => { const d = e.target.closest(".row"); if (d) open(+d.dataset.i); });
["filter", "ffont", "fstat", "fsort"].forEach(id => $(id).addEventListener("input", renderList));

async function loadRows(keep) {
  const j = await api("/api/glyphs");
  rows = j.rows; hasReview = !!j.review;
  $("src").textContent = (j.review ? j.review.split("/").pop() : "all generated glyphs") + " · " + rows.length + " glyphs";
  $("acc").disabled = $("rej").disabled = !hasReview;
  if (!$("ffont").options.length) {
    const fonts = [...new Set(rows.map(r => r.font))].sort();
    $("ffont").innerHTML = '<option value="">All fonts</option>' + fonts.map(f => `<option>${f}</option>`).join("");
  }
  renderList();
  if (!keep && rows.length) open(visible()[0] ?? 0);
}
async function open(i) {
  if (dirty && !confirm("Discard unsaved changes to this glyph?")) return;
  idx = i; const r = rows[i];
  glyph = await api(`/api/glyph?font=${encodeURIComponent(r.font)}&cp=${encodeURIComponent(r.codepoint)}`);
  wedges = glyph.wedges.map(w => ({...w, apex: [...w.apex], head_a: [...w.head_a], head_b: [...w.head_b], tail: [...w.tail]}));
  sel = -1; dirty = false; undoStack = []; $("notes").value = r.notes || ""; msg("");
  img = new Image();
  img.onload = () => { fit(); draw(); };
  img.src = `/image?font=${encodeURIComponent(r.font)}&cp=${encodeURIComponent(r.codepoint)}&t=${Date.now()}`;
  renderList(); renderWedges();
  const el = document.querySelector(".row.sel"); if (el) el.scrollIntoView({block: "nearest"});
}
function renderWedges() {
  $("wl").innerHTML = wedges.map((w, i) => `<li data-i="${i}" class="${i === sel ? "sel" : ""}"><span>Wedge ${i + 1}</span><span class="${w.confidence !== null && w.confidence < 0.3 ? "w" : ""}">${w.confidence === null || w.confidence === undefined ? "edited" : "conf " + w.confidence.toFixed(2)}${w.off_ink ? " · off ink" : ""}${w.rare ? " · rare" : ""}</span></li>`).join("") || '<li>No wedges</li>';
  $("del").disabled = $("swa").disabled = $("swb").disabled = sel < 0;
  $("undo").disabled = !undoStack.length;
  $("save").textContent = dirty ? "Save correction *" : "Save correction";
}
$("wl").addEventListener("click", e => { const li = e.target.closest("li[data-i]"); if (li) { sel = +li.dataset.i; renderWedges(); draw(); } });

function resize() { const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = r.width * d; cv.height = r.height * d; ctx.setTransform(d, 0, 0, d, 0, 0); draw(); }
function fit() { if (!glyph) return; const r = cv.getBoundingClientRect(); const k = Math.min((r.width - 60) / glyph.width, (r.height - 60) / glyph.height); view = {k, ox: (r.width - glyph.width * k) / 2, oy: (r.height - glyph.height * k) / 2}; }
const toS = p => [view.ox + (p[0] + 0.5) * view.k, view.oy + (p[1] + 0.5) * view.k];
const toI = (x, y) => [(x - view.ox) / view.k - 0.5, (y - view.oy) / view.k - 0.5];

function draw() {
  const r = cv.getBoundingClientRect(); ctx.clearRect(0, 0, r.width, r.height);
  if (!glyph || !img || !img.complete) return;
  ctx.imageSmoothingEnabled = view.k < 3;
  ctx.globalAlpha = 0.55; ctx.drawImage(img, view.ox, view.oy, glyph.width * view.k, glyph.height * view.k); ctx.globalAlpha = 1;
  ctx.strokeStyle = "#e1e0d9"; ctx.lineWidth = 1; ctx.strokeRect(view.ox, view.oy, glyph.width * view.k, glyph.height * view.k);
  wedges.forEach((w, i) => {
    const doubtful = w.confidence !== null && w.confidence !== undefined && w.confidence < 0.3;
    const col = i === sel ? "#0b0b0b" : (doubtful ? "#eb6834" : "#2a78d6");
    const a = toS(w.apex), ha = toS(w.head_a), hb = toS(w.head_b), t = toS(w.tail);
    ctx.lineWidth = i === sel ? 2.5 : 1.75; ctx.strokeStyle = col; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(ha[0], ha[1]); ctx.lineTo(hb[0], hb[1]); ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(t[0], t[1]); ctx.stroke();
    const R = i === sel ? 6 : 4.5;
    ctx.fillStyle = "#fff"; ctx.lineWidth = 2;
    for (const h of [ha, hb]) { ctx.fillStyle = "#1baf7a"; ctx.strokeStyle = "#fff"; ctx.fillRect(h[0] - R, h[1] - R, 2 * R, 2 * R); ctx.strokeRect(h[0] - R, h[1] - R, 2 * R, 2 * R); }
    ctx.beginPath(); ctx.arc(t[0], t[1], R, 0, 7); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#4a3aa7"; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(a[0], a[1], R + 1, 0, 7); ctx.fillStyle = doubtful && i !== sel ? "#eb6834" : "#2a78d6"; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#0b0b0b"; ctx.font = "11px system-ui"; ctx.fillText(String(i + 1), a[0] + 9, a[1] - 8);
  });
  $("hud").textContent = `${glyph.font} ${glyph.codepoint} ${glyph.name || ""} · ${glyph.width}×${glyph.height} px · ${wedges.length} wedges · source: ${glyph.source}${dirty ? " · unsaved" : ""}`;
}
function hit(x, y) {
  let best = null;
  wedges.forEach((w, i) => PTS.forEach(p => { const s = toS(w[p]); const d = Math.hypot(s[0] - x, s[1] - y); if (d < 12 && (!best || d < best.d)) best = {i, p, d}; }));
  return best;
}
function snapshot() { undoStack.push(JSON.stringify(wedges)); if (undoStack.length > 100) undoStack.shift(); }
function touch() { dirty = true; renderWedges(); draw(); }

cv.addEventListener("pointerdown", e => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, h = hit(x, y);
  cv.setPointerCapture(e.pointerId);
  if (h) { snapshot(); sel = h.i; drag = {kind: "pt", ...h, whole: e.shiftKey && h.p === "apex", last: toI(x, y)}; renderWedges(); draw(); }
  else { drag = {kind: "pan", x, y, ox: view.ox, oy: view.oy}; }
});
cv.addEventListener("pointermove", e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (drag.kind === "pan") { view.ox = drag.ox + x - drag.x; view.oy = drag.oy + y - drag.y; draw(); return; }
  const p = toI(x, y), w = wedges[drag.i];
  const clamp = q => [Math.max(0, Math.min(glyph.width - 1, q[0])), Math.max(0, Math.min(glyph.height - 1, q[1]))];
  if (drag.whole) { const dx = p[0] - drag.last[0], dy = p[1] - drag.last[1]; PTS.forEach(k => { w[k] = clamp([w[k][0] + dx, w[k][1] + dy]); }); drag.last = p; }
  else w[drag.p] = clamp(p);
  w.confidence = null; w.off_ink = false; w.rare = false; dirty = true; draw();
});
cv.addEventListener("pointerup", () => { if (drag && drag.kind === "pt") renderWedges(); drag = null; });
cv.addEventListener("wheel", e => {
  e.preventDefault(); const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const f = Math.exp(-e.deltaY * 0.0015), k = Math.max(0.05, Math.min(60, view.k * f));
  view.ox = x - (x - view.ox) * (k / view.k); view.oy = y - (y - view.oy) * (k / view.k); view.k = k; draw();
}, {passive: false});

function addWedge() {
  if (!glyph) return; snapshot();
  const r = cv.getBoundingClientRect(), c = toI(r.width / 2, r.height / 2), u = Math.max(glyph.width, glyph.height) / 256;
  wedges.push({apex: c, head_a: [c[0] - 10 * u, c[1] - 9 * u], head_b: [c[0] - 10 * u, c[1] + 9 * u], tail: [c[0] + 40 * u, c[1]], confidence: null, off_ink: false, rare: false});
  sel = wedges.length - 1; touch();
}
function delWedge() { if (sel < 0) return; snapshot(); wedges.splice(sel, 1); sel = -1; touch(); }
function swap(k) { if (sel < 0) return; snapshot(); const w = wedges[sel], t = w.tail; w.tail = w[k]; w[k] = t; w.confidence = null; w.rare = false; w.off_ink = false; touch(); }
function undo() { if (!undoStack.length) return; wedges = JSON.parse(undoStack.pop()); sel = Math.min(sel, wedges.length - 1); touch(); }
async function save() {
  if (!glyph) return;
  try {
    const j = await api("/api/save", {font: glyph.font, codepoint: glyph.codepoint, wedges: wedges.map(w => ({apex: w.apex, head_a: w.head_a, head_b: w.head_b, tail: w.tail}))});
    dirty = false; glyph.source = "reviewed"; msg(`Saved ${j.wedges} wedges to ${j.saved}`, "ok");
    await loadRows(true); renderWedges(); draw();
  } catch (err) { msg(err.message, "err"); }
}
async function review(accept) {
  if (!glyph || !hasReview) return;
  try { await api("/api/review", {font: glyph.font, codepoint: glyph.codepoint, accept, notes: $("notes").value}); msg(accept === "y" ? "Marked accepted" : "Marked rejected", "ok"); const v = visible(), p = v.indexOf(idx), nxt = v[p + 1]; await loadRows(true); if (nxt !== undefined) open(nxt); }
  catch (err) { msg(err.message, "err"); }
}
async function revert() {
  if (!glyph || !confirm("Delete the saved correction for this glyph and go back to the generated skeleton?")) return;
  await api("/api/revert", {font: glyph.font, codepoint: glyph.codepoint}); dirty = false; await loadRows(true); open(idx);
}
function go(d) { const v = visible(), p = v.indexOf(idx), n = p < 0 ? v[0] : v[p + d]; if (n !== undefined) open(n); }
function nextOpen() { const v = visible(), p = v.indexOf(idx); const n = v.slice(p + 1).concat(v.slice(0, Math.max(p, 0))).find(i => statusOf(rows[i]) === "open"); if (n !== undefined) open(n); else msg("No open glyphs in this view", "ok"); }

$("add").onclick = addWedge; $("del").onclick = delWedge; $("undo").onclick = undo; $("save").onclick = save;
$("swa").onclick = () => swap("head_a"); $("swb").onclick = () => swap("head_b");
$("fit").onclick = () => { fit(); draw(); }; $("prev").onclick = () => go(-1); $("next").onclick = () => go(1); $("nopen").onclick = nextOpen;
$("acc").onclick = () => review("y"); $("rej").onclick = () => review("n"); $("revert").onclick = revert;
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  else if (e.key === "Delete" || e.key === "Backspace") delWedge();
  else if (e.key.toLowerCase() === "a") addWedge();
  else if (e.key.toLowerCase() === "f") { fit(); draw(); }
  else if (e.key.toLowerCase() === "n") nextOpen();
  else if (e.key.toLowerCase() === "y") review("y");
  else if (e.key.toLowerCase() === "x") review("n");
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
});
window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
window.addEventListener("resize", resize);
resize(); loadRows(false).catch(err => msg(err.message, "err"));
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
