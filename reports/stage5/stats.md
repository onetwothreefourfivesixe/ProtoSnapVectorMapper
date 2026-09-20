# Stage 5 report: batch inference and review queue

Model: hybrid on `checkpoints/retrain/plus_s1/best.pt`. Output: `generated/<Font>/<hex>_adf.csv` and `_con.csv`, same format as `skeletons/`, in original-image coordinates. `skeletons/` is not modified.

## What was processed

| Font | Unlabelled images | Blank (skipped) | Copied from an identical human-labelled image | Predicted by the model | Model found no wedges |
|---|---|---|---|---|---|
| Assurbanipal | 719 | 177 | 6 | 536 | 0 |
| Esagil | 370 | 0 | 0 | 370 | 0 |
| Santakku | 634 | 627 | 7 | 0 | 0 |

## Verification

919 written skeletons were reloaded through the normal loader. Failures: 0. Largest model-to-original-to-model round-trip error: 1.14e-13 px.

## Confidence of model predictions

![confidence](../charts/stage5_confidence.png)

| Font | Glyphs | Median confidence | Below 0.3 | 0.3 to 0.6 | Above 0.6 | Glyphs with a rare-orientation wedge | Glyphs with an off-ink tail | Glyphs with a point clamped to the image |
|---|---|---|---|---|---|---|---|---|
| Assurbanipal | 536 | 0.04 | 343 | 90 | 103 | 97 | 121 | 3 |
| Esagil | 370 | 0.57 | 142 | 54 | 174 | 30 | 36 | 35 |

## How much correction work there is

Glyph confidence is the weakest wedge, and these glyphs are complex (many have more than 10 wedges), so most glyphs score near zero even when only one wedge is in doubt. The practical number is how many wedges per glyph fall below 0.3 confidence.

| Font | Wedges predicted | Doubtful wedges | Glyphs with 0 doubtful | 1 | 2 | 3 or more |
|---|---|---|---|---|---|---|
| Assurbanipal | 5775 | 693 (12%) | 193 | 165 | 83 | 95 |
| Esagil | 2480 | 225 (9%) | 228 | 87 | 36 | 19 |

## What the confidence means (calibration on 90 labelled val + test glyphs)

| Confidence band | Labelled glyphs | Fully correct within 5 px |
|---|---|---|
| 0.0 to 0.3 | 40 | 0.20 |
| 0.3 to 0.6 | 9 | 0.67 |
| 0.6 to 1.0 | 41 | 0.73 |

Applied to the 536 Assurbanipal and Santakku predictions, this suggests roughly 204 (38%) need no correction at all. Esagil is a different font the model never saw, so this calibration does not transfer to it; its acceptance rate has to come from the review pack.

## Review packs

Fill the `accept` column (y or n) and optionally `notes`, then run `python -m protosnap.stage5 acceptance <csv>`. The random pack estimates the overall acceptance rate; the lowest-confidence pack is where correction effort pays most.

| Pack | Glyphs | Mean confidence | Review sheet | Contact sheets |
|---|---|---|---|---|
| assurbanipal_santakku_lowest | 48 | 0.00 | `reports/stage5/review/review_assurbanipal_santakku_lowest.csv` | `reports/stage5/review/assurbanipal_santakku_lowest_01.png`, `reports/stage5/review/assurbanipal_santakku_lowest_02.png`, `reports/stage5/review/assurbanipal_santakku_lowest_03.png`, `reports/stage5/review/assurbanipal_santakku_lowest_04.png` |
| assurbanipal_santakku_random | 48 | 0.20 | `reports/stage5/review/review_assurbanipal_santakku_random.csv` | `reports/stage5/review/assurbanipal_santakku_random_01.png`, `reports/stage5/review/assurbanipal_santakku_random_02.png`, `reports/stage5/review/assurbanipal_santakku_random_03.png`, `reports/stage5/review/assurbanipal_santakku_random_04.png` |
| esagil_lowest | 48 | 0.00 | `reports/stage5/review/review_esagil_lowest.csv` | `reports/stage5/review/esagil_lowest_01.png`, `reports/stage5/review/esagil_lowest_02.png`, `reports/stage5/review/esagil_lowest_03.png`, `reports/stage5/review/esagil_lowest_04.png` |
| esagil_random | 48 | 0.46 | `reports/stage5/review/review_esagil_random.csv` | `reports/stage5/review/esagil_random_01.png`, `reports/stage5/review/esagil_random_02.png`, `reports/stage5/review/esagil_random_03.png`, `reports/stage5/review/esagil_random_04.png` |
