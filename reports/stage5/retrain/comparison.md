# Retraining comparison

Every checkpoint is scored as the shipped hybrid on the frozen validation and test sets (original human annotations only). Values are means over seeds with the range in brackets. One glyph is 2.2 points of exact-match on a 45-glyph split, so differences smaller than the seed range are not evidence.

![retrain](../../charts/stage5_retrain.png)

## Test split

| Model | Seeds | F1 | Corner err (px) | Heads | Tail | Exact-match |
|---|---|---|---|---|---|---|
| Stage 4 model (old targets, original data) | 1 | 0.980 | 2.18 | 2.11 | 3.08 | 0.644 |
| Fixed targets, original data | 2 | 0.970 (0.970 to 0.970) | 2.25 (2.150 to 2.351) | 2.23 | 3.10 (3.007 to 3.185) | 0.589 (0.533 to 0.644) |
| Fixed targets, original + your reviewed labels | 2 | 0.980 (0.979 to 0.980) | 2.13 (2.085 to 2.165) | 2.06 | 2.94 (2.807 to 3.075) | 0.656 (0.622 to 0.689) |

## Val split

| Model | Seeds | F1 | Corner err (px) | Heads | Tail | Exact-match |
|---|---|---|---|---|---|---|
| Stage 4 model (old targets, original data) | 1 | 0.986 | 2.98 | 3.06 | 4.29 | 0.689 |
| Fixed targets, original data | 2 | 0.985 (0.979 to 0.990) | 2.74 (2.461 to 3.023) | 2.74 | 4.03 (3.476 to 4.580) | 0.689 (0.667 to 0.711) |
| Fixed targets, original + your reviewed labels | 2 | 0.988 (0.988 to 0.988) | 2.81 (2.799 to 2.830) | 2.82 | 4.11 (4.016 to 4.212) | 0.689 (0.667 to 0.711) |

## Test split per font

| Model | Font | F1 | Corner err | Tail | Exact-match |
|---|---|---|---|---|---|
| Stage 4 model (old targets, original data) | Assurbanipal | 0.996 | 1.86 | 2.63 | 0.688 |
| Stage 4 model (old targets, original data) | Santakku | 0.971 | 2.37 | 3.36 | 0.621 |
| Fixed targets, original data | Assurbanipal | 0.985 | 1.96 | 2.66 | 0.594 |
| Fixed targets, original data | Santakku | 0.962 | 2.43 | 3.36 | 0.586 |
| Fixed targets, original + your reviewed labels | Assurbanipal | 0.996 | 1.76 | 2.43 | 0.750 |
| Fixed targets, original + your reviewed labels | Santakku | 0.970 | 2.35 | 3.25 | 0.603 |

## Failure classes on test (mean count over seeds)

| Failure class | Stage 4 model (old targets, original data) | Fixed targets, original data | Fixed targets, original + your reviewed labels |
|---|---|---|---|
| head tip off by >10 px | 0.0 | 4.5 | 1.0 |
| head tip off by >10 px (rare orientation) | 0.0 | 1.5 | 0.5 |
| prong and tail swapped | 12.0 | 9.5 | 10.0 |
| tail stopped early (>10 px short) | 6.0 | 7.5 | 6.0 |
| tail overshoot or sideways (>10 px) | 2.0 | 2.5 | 3.0 |
| tail off by >10 px (rare orientation) | 11.0 | 9.0 | 10.0 |
| near miss: FN with a prediction 10-20 px away | 2.0 | 3.5 | 1.0 |
| near miss: FP within 10-20 px of a wedge | 3.0 | 6.5 | 5.5 |
| missed wedge (FN, nothing nearby) | 0.0 | 1.0 | 0.0 |
| spurious wedge (FP, nothing nearby) | 7.0 | 7.0 | 6.0 |

## Proxies on still-unreviewed glyphs (no ground truth)

Off-ink tails are a model-independent validity check; doubtful wedges are the model's own confidence below 0.3. Lower is better for both.

| Model | Assurbanipal: off-ink tails | Assurbanipal: doubtful wedges | Esagil: off-ink tails | Esagil: doubtful wedges |
|---|---|---|---|---|
| Stage 4 model (old targets, original data) | 3.9% | 19.9% | 0.8% | 14.1% |
| Fixed targets, original data | 6.1% | 19.0% | 2.1% | 14.0% |
| Fixed targets, original + your reviewed labels | 2.6% | 10.6% | 1.2% | 7.9% |
