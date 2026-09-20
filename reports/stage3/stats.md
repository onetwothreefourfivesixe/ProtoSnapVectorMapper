# Stage 3 report: learned keypoint model

Match threshold 10 px. Metric definitions as in Stage 1. Model selection and decode tuning use the validation split only; the test split is scored once per checkpoint.

## Runs

| Run | Encoder | Extra channels | Params (M) | Epochs | Best epoch | Train time (min) | Decode |
|---|---|---|---|---|---|---|---|
| resnet34 | resnet34 | False | 24.6 | 60 | 36 | 42.6 | thr 0.3, snap 5.0 |
| unet | unet | False | 7.9 | 60 | 42 | 84.6 | thr 0.2, snap 5.0 |
| unet_xc | unet | True | 7.9 | 60 | 46 | 20.5 | thr 0.3, snap 5.0 |

## Charts

![metrics_test](../charts/metrics_test.png)

![metrics_by_font_test](../charts/metrics_by_font_test.png)

![failure_classes_test](../charts/failure_classes_test.png)


## Scores

| Predictor | Split | Font | Glyphs | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Corner err (px) | Apex | Head a | Head b | Tail | Exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | val | all | 45 | 256 | 259 | 250 | 9 | 6 | 0.965 | 0.977 | 0.971 | 4.32 | 1.73 | 4.83 | 4.27 | 6.44 | 0.533 |
| classical | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 2.48 | 1.68 | 3.00 | 2.55 | 2.70 | 0.688 |
| classical | val | Santakku | 29 | 177 | 180 | 171 | 9 | 6 | 0.950 | 0.966 | 0.958 | 5.16 | 1.75 | 5.68 | 5.06 | 8.16 | 0.448 |
| resnet34 | val | all | 45 | 256 | 260 | 254 | 6 | 2 | 0.977 | 0.992 | 0.984 | 2.86 | 1.42 | 2.48 | 2.82 | 4.71 | 0.644 |
| resnet34 | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 2.32 | 1.61 | 1.75 | 2.19 | 3.72 | 0.750 |
| resnet34 | val | Santakku | 29 | 177 | 181 | 175 | 6 | 2 | 0.967 | 0.989 | 0.978 | 3.10 | 1.34 | 2.80 | 3.11 | 5.15 | 0.586 |
| unet | val | all | 45 | 256 | 260 | 255 | 5 | 1 | 0.981 | 0.996 | 0.988 | 3.05 | 1.45 | 2.69 | 2.95 | 5.11 | 0.578 |
| unet | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 2.45 | 1.55 | 1.90 | 2.44 | 3.91 | 0.688 |
| unet | val | Santakku | 29 | 177 | 181 | 176 | 5 | 1 | 0.972 | 0.994 | 0.983 | 3.32 | 1.41 | 3.05 | 3.17 | 5.65 | 0.517 |
| unet_xc | val | all | 45 | 256 | 259 | 254 | 5 | 2 | 0.981 | 0.992 | 0.986 | 3.11 | 1.50 | 2.92 | 3.20 | 4.82 | 0.622 |
| unet_xc | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 2.44 | 1.58 | 2.37 | 2.47 | 3.35 | 0.688 |
| unet_xc | val | Santakku | 29 | 177 | 180 | 175 | 5 | 2 | 0.972 | 0.989 | 0.980 | 3.41 | 1.47 | 3.17 | 3.53 | 5.48 | 0.586 |
| classical | test | all | 45 | 300 | 308 | 291 | 17 | 9 | 0.945 | 0.970 | 0.957 | 3.78 | 1.60 | 3.83 | 3.65 | 6.04 | 0.356 |
| classical | test | Assurbanipal | 16 | 114 | 114 | 112 | 2 | 2 | 0.982 | 0.982 | 0.982 | 3.29 | 1.70 | 3.26 | 3.13 | 5.09 | 0.375 |
| classical | test | Santakku | 29 | 186 | 194 | 179 | 15 | 7 | 0.923 | 0.962 | 0.942 | 4.08 | 1.53 | 4.18 | 3.97 | 6.64 | 0.345 |
| resnet34 | test | all | 45 | 300 | 311 | 297 | 14 | 3 | 0.955 | 0.990 | 0.972 | 2.45 | 1.43 | 2.01 | 2.23 | 4.14 | 0.511 |
| resnet34 | test | Assurbanipal | 16 | 114 | 112 | 111 | 1 | 3 | 0.991 | 0.974 | 0.982 | 2.17 | 1.48 | 1.52 | 1.56 | 4.10 | 0.438 |
| resnet34 | test | Santakku | 29 | 186 | 199 | 186 | 13 | 0 | 0.935 | 1.000 | 0.966 | 2.62 | 1.40 | 2.30 | 2.62 | 4.16 | 0.552 |
| unet | test | all | 45 | 300 | 310 | 297 | 13 | 3 | 0.958 | 0.990 | 0.974 | 2.49 | 1.40 | 1.93 | 2.49 | 4.12 | 0.467 |
| unet | test | Assurbanipal | 16 | 114 | 114 | 113 | 1 | 1 | 0.991 | 0.991 | 0.991 | 2.33 | 1.49 | 1.60 | 1.98 | 4.26 | 0.438 |
| unet | test | Santakku | 29 | 186 | 196 | 184 | 12 | 2 | 0.939 | 0.989 | 0.963 | 2.58 | 1.35 | 2.14 | 2.79 | 4.04 | 0.483 |
| unet_xc | test | all | 45 | 300 | 308 | 298 | 10 | 2 | 0.968 | 0.993 | 0.980 | 2.33 | 1.39 | 2.08 | 2.15 | 3.71 | 0.511 |
| unet_xc | test | Assurbanipal | 16 | 114 | 113 | 113 | 0 | 1 | 1.000 | 0.991 | 0.996 | 2.11 | 1.44 | 1.79 | 1.60 | 3.59 | 0.438 |
| unet_xc | test | Santakku | 29 | 186 | 195 | 185 | 10 | 1 | 0.949 | 0.995 | 0.971 | 2.47 | 1.37 | 2.26 | 2.48 | 3.78 | 0.552 |

## Review gate: `resnet34` (best on validation) versus the classical baseline on test

| Font | Classical F1 | Learned F1 | Classical err | Learned err | Classical exact | Learned exact | Learned better? |
|---|---|---|---|---|---|---|---|
| all | 0.957 | 0.972 | 3.78 | 2.45 | 0.356 | 0.511 | yes |
| Assurbanipal | 0.982 | 0.982 | 3.29 | 2.17 | 0.375 | 0.438 | yes |
| Santakku | 0.942 | 0.966 | 4.08 | 2.62 | 0.345 | 0.552 | yes |

Gate (learned beats classical on the combined objective for both fonts): **PASSED**.

## Failure classes on test

| Failure class | classical | resnet34 |
|---|---|---|
| head tip off by >10 px | 27 | 1 |
| head tip off by >10 px (rare orientation) | 4 | 0 |
| prong and tail swapped | 6 | 12 |
| tail stopped early (>10 px short) | 12 | 11 |
| tail overshoot or sideways (>10 px) | 16 | 15 |
| tail off by >10 px (rare orientation) | 9 | 9 |
| near miss: FN with a prediction 10-20 px away | 8 | 2 |
| near miss: FP within 10-20 px of a wedge | 10 | 8 |
| missed wedge (FN, nothing nearby) | 1 | 1 |
| spurious wedge (FP, nothing nearby) | 7 | 6 |

Curves: `reports/stage3/resnet34/curves.png`, `reports/stage3/unet/curves.png`, `reports/stage3/unet_xc/curves.png`
Failure gallery of `resnet34` on test: `reports/stage3/gallery/resnet34_test_01.png`, `reports/stage3/gallery/resnet34_test_02.png`

