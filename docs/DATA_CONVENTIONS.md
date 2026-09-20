# Data conventions

Everything a new contributor, or a downstream consumer such as ProtoSnap, needs to know about the files.

## Folders

| Path | Who writes it | Contents |
|---|---|---|
| `prototypes/<Font>/<range>/<hex>.png` | upstream | One rendered glyph per Unicode codepoint. Assurbanipal and Santakku are 256x256 px. Esagil is 512 px high and variable width. Ink is dark on white, drawn as an outline about 2 to 3 px thick at the 256 px scale. |
| `prototypes/metadata.csv` | upstream | One row per codepoint: which fonts contain the sign, and its name. |
| `skeletons/<Font>/<hex>_adf.csv`, `_con.csv` | annotators, and `protosnap promote` | The labelled set. Original hand annotations plus skeletons promoted after review. |
| `data/promoted.csv` | `protosnap promote` | Provenance: which files under `skeletons/` came from reviewed model output, whether they were corrected by hand, and the date. |
| `data/corrections/<Font>/` | `protosnap correct` | Repairs to original annotations. Read instead of the original; the original is never modified. |
| `data/reviewed/<Font>/` | the editor | Hand corrections of generated skeletons, waiting to be promoted. |
| `data/splits.json` | `protosnap stage0`, `promote` | Codepoint lists for train, validation and test. |
| `generated/<Font>/` | `protosnap infer` | Machine skeletons awaiting review, with `manifest.csv` (one row per glyph, in review order) and `wedges.csv` (one row per wedge). |
| `models/` | maintainers | Shipped weights and `expected_scores.json` for `protosnap reproduce`. |
| `reports/` | the stage runners | Reports, charts, galleries, review packs. |

Codepoints are lowercase hexadecimal with a `0x` prefix, for example `0x12000`.

## A skeleton

A skeleton is a set of strokes. Each stroke is a wedge with four points, which is ProtoSnap's four-keypoint scheme: three keypoints for the corners of the triangular head and one for the end of the tail.

```
   head ●─────────● head          The apex is the head corner the tail leaves from.
         \       /                Edges: apex–head, apex–head, head–head, apex–tail.
          \     /
           ● apex
           │
           │
           ● tail
```

Coordinates are pixels in the glyph image, x to the right and y down, with a pixel's centre at integer coordinates. For Esagil they are pixels of the original full-size image, not of the 256 px model canvas.

### The two files

`<hex>_adf.csv` lists points, `<hex>_con.csv` lists edges between point indices *within* a stroke. Two layouts occur in the original annotations and both are read transparently (columns are found by header name):

| | Layout A | Layout B |
|---|---|---|
| Original files | all 288 Santakku, 70 Assurbanipal | 91 Assurbanipal |
| Point file header | `label,x,y` | `x,y,label` |
| Edge file header | `label,i,j` | `i,j,label` |
| Coordinates | integers | full-precision floats |
| Edge rows per stroke | 2-0, 2-1, 2-3, 0-1 | 0-1, 0-2, 1-2, 2-3 |

Shared by both: CRLF line endings, a final newline, labels `Stroke 1`, `Stroke 2`, … , and points in the order head, head, apex, tail, so the apex is index 2 and the tail index 3.

Generated and corrected files are written in the dominant layout of their font: Santakku and Esagil in A, Assurbanipal in B. Strokes are numbered left to right by apex, and the upper head is listed first (the left one on ties). Those two orderings are the majority habit in the originals, 71% and 67%, not a strict rule there. `protosnap format check <dir>` reports files that deviate.

### Do not rely on point order when reading

The originals are not consistent: the apex is index 2 in 91% of strokes but index 1, 3 or 0 in the rest. Find the apex by topology, as `protosnap.data.classify_stroke` does: it is the point with three edges, the tail is the point with one. The loader returns a canonical `Wedge(apex, head_a, head_b, tail)` whose head order is fixed by a cross-product sign and is independent of file order.

### Strokes that are not wedges

28 original strokes are closed three-point triangles with no tail, and one has a self-loop edge. They are loaded with `kind="triangle"` or `"malformed"`, excluded from `Skeleton.wedges`, ignored by the training loss within 14 px, and skipped by the secondary metric. Generated files never contain them.

## Labelled versus promoted

`find_pairs(root)` returns only the original 449 hand-annotated pairs. `find_pairs(root, include_promoted=True)` adds skeletons promoted from reviewed model output, marked `Pair.promoted`, possibly in Esagil. Three rules keep evaluation honest:

1. Promoted labels are never part of validation or test. Most were accepted as the model produced them, so scoring against them would grade the model on its own output.
2. A promoted glyph is trained on only if its codepoint is in the train split. The split is by codepoint, so a sign held out in one font is held out in every font.
3. New codepoints are added to train only. Validation and test stay 45 glyphs each, so results remain comparable across rounds.

## Images at other sizes

`protosnap.imageprep.prepare_image` brings any glyph to the 256 px canvas the models expect. 256x256 images pass through. Anything else is area-resampled so a 512 px em height becomes 80 px (wide signs are scaled further to fit 240 px), then centred. It returns a `Transform` with `to_model` and `to_original`, an exact inverse pair, so skeletons stay in original-image coordinates on disk.

## Known oddities in the source data

- 627 Santakku images are blank because the font lacks those signs; the metadata says so.
- 177 Assurbanipal images are blank although the metadata says the font has the sign. Rendering failed upstream; see `reports/stage5/blank_images.csv`.
- The fonts share outlines: 247 codepoints have byte-identical images in Assurbanipal and Santakku, and 70 labelled pairs are exact duplicates across the two fonts. Only 379 of the 449 pairs are distinct.
- HI (`0x1212d`) and SHAR2 (`0x122b9`) are pixel-identical in Assurbanipal with different skeletons; SHAR2 is pinned to train.
- HI TIMES BAD (Santakku `0x12130`) was annotated without apexes; `data/corrections/` holds the repair.
