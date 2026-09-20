# Evaluation metric

Defined in Stage 1 and frozen since. Implemented in `protosnap/metrics.py`; every reported number in this repository comes from it.

## What is compared

A prediction for a glyph is a list of wedges. The ground truth is the list of canonical wedges in the glyph's skeleton. Strokes that are not wedges (tail-less triangles, one malformed stroke) are not part of the ground truth.

## Matching

Predicted and ground-truth wedges are paired by Hungarian assignment on the distance between apexes. An assigned pair counts as a **match** only if that distance is at most the match threshold, **10 px**. The median wedge head is about 21 px wide, so this is roughly half a head.

## Numbers reported

| Number | Definition |
|---|---|
| Precision, recall, F1 | Micro-averaged over all wedges in the split: matched / predicted, matched / ground truth. |
| Corner error | For matched wedges, the mean Euclidean distance over the four points (apex, head, head, tail). Also reported per point type. The two heads are compared in canonical order, which does not depend on file order. |
| Exact-match | The share of glyphs that are entirely right: no unmatched prediction, no unmatched ground-truth wedge, and every point of every matched wedge within the match threshold. This is the strict headline number. |

Everything is reported per split and per font. Validation and test hold 45 glyphs each, so **one glyph is 2.2 points of exact-match**. Differences smaller than that, or smaller than the spread between training seeds, are not evidence of anything; `protosnap compare` exists to show the seed spread next to the difference.

## Reference points

| Predictor | F1 | Corner error | Exact-match | |
|---|---|---|---|---|
| Null (predicts nothing) | 0.000 | n/a | 0.000 | the floor |
| Oracle (ground truth) | 1.000 | 0.00 px | 1.000 | the ceiling, proves the harness agrees with itself |
| Jittered oracle | 0.903 | 2.49 px | 0.333 | ground truth with 2 px noise, 10% of wedges dropped, 5% spurious |

The jittered oracle is the useful one for intuition: 2 px of noise per point alone costs two thirds of exact-match.

## Edge cases

- A glyph whose strokes are all triangles has no evaluable wedges, so predicting nothing for it is exact. MUSH (Santakku `0x12232`) is the one such glyph, in train.
- Precision is reported as 0, not undefined, when nothing is predicted.
- **Detections on unlabelled heads.** Triangle strokes are real wedge heads in the image without a wedge label, so a correct detection there counts as a false positive under the frozen metric. `evaluate(..., ignore_non_wedge=14)` is a secondary metric that, *after* matching, skips unmatched predictions within 14 px of a non-wedge stroke. Matched predictions are never skipped. It is reported next to the frozen metric, never instead of it.

## Model selection objective

Checkpoint selection, decode tuning and the classical tuner all maximise `F1 + exact-match − corner error / 100` on validation, so one pixel of mean error trades against one point of F1. Test is scored once per configuration and never used for selection.

## Failure classes

`protosnap/analysis.py` sorts errors into named classes (head tip off by more than 10 px, tail stopped early, tail overshoot or sideways, prong and tail swapped, rare orientation, near-miss pairs, missed, spurious) with the same definitions for every model, so the classical, learned and hybrid models are described in the same terms.

## Confidence is not accuracy

A wedge's confidence is the weakest of the network's apex score and its heatmap support at the tail end and both head tips, times 0.6 when the tail points outside the usual arc; a tail that leaves the ink caps it at 0.05. A glyph's confidence is its least certain wedge. It ranks correctness (AUROC about 0.87 per glyph on held-out data) and orders the review queue. It is the model's own estimate: after retraining, the top band became less precise (73% fully correct, down from 92%), and a reviewer accepted about 90% of the *lowest*-confidence glyphs, so it is much stricter than human acceptability.
