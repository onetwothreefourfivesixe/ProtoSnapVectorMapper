# Stage 2 report: classical baseline

Parameters: tuned on val (`reports/stage2/tuned_params.json`). Match threshold 10 px. Metric definitions as in Stage 1.

```
{
 "ink_threshold": 128,
 "max_hole_area": 24,
 "head_radius": 4.0,
 "head_radius_scale": 0.8,
 "side_max_len": 20.0,
 "opening_weight": 0.5,
 "no_tail_penalty": 15.0,
 "tail_max_dev": 45.0,
 "missing_prong_penalty": 8.0,
 "prong_dev_weight": 0.0,
 "prong_dir_from_centroid": true,
 "tail_dir_weight": 4.0,
 "tail_arc": [
  -40.0,
  130.0
 ],
 "prong_dir_weight": 4.0,
 "prong_forbidden_arc": [
  -20.0,
  120.0
 ],
 "capped_prong_penalty": 50.0,
 "prong_max_dev": 50.0,
 "prong_max_len": 16.0,
 "prong_default_len": 6.0,
 "tip_extension": 0.0,
 "apex_shift": 1.0,
 "continue_angle": 35.0,
 "max_tail_hops": 8,
 "max_prong_hops": 1,
 "stop_at_heads": true,
 "pass_through_heads": true,
 "merge_angle": 60.0,
 "nms_apex_dist": 3.0,
 "nms_angle": 90.0
}
```

## Scores

| Predictor | Split | Font | Glyphs | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Corner err (px) | Apex | Head a | Head b | Tail | Exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | train | all | 359 | 2170 | 2186 | 2117 | 69 | 53 | 0.968 | 0.976 | 0.972 | 4.54 | 1.66 | 4.88 | 4.28 | 7.36 | 0.287 |
| classical | train | Assurbanipal | 129 | 733 | 721 | 711 | 10 | 22 | 0.986 | 0.970 | 0.978 | 3.97 | 1.69 | 4.20 | 3.54 | 6.46 | 0.349 |
| classical | train | Santakku | 230 | 1437 | 1465 | 1406 | 59 | 31 | 0.960 | 0.978 | 0.969 | 4.83 | 1.64 | 5.23 | 4.65 | 7.82 | 0.252 |
| classical | val | all | 45 | 256 | 259 | 250 | 9 | 6 | 0.965 | 0.977 | 0.971 | 4.32 | 1.73 | 4.83 | 4.27 | 6.44 | 0.533 |
| classical | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 2.48 | 1.68 | 3.00 | 2.55 | 2.70 | 0.688 |
| classical | val | Santakku | 29 | 177 | 180 | 171 | 9 | 6 | 0.950 | 0.966 | 0.958 | 5.16 | 1.75 | 5.68 | 5.06 | 8.16 | 0.448 |
| classical | test | all | 45 | 300 | 308 | 291 | 17 | 9 | 0.945 | 0.970 | 0.957 | 3.78 | 1.60 | 3.83 | 3.65 | 6.04 | 0.356 |
| classical | test | Assurbanipal | 16 | 114 | 114 | 112 | 2 | 2 | 0.982 | 0.982 | 0.982 | 3.29 | 1.70 | 3.26 | 3.13 | 5.09 | 0.375 |
| classical | test | Santakku | 29 | 186 | 194 | 179 | 15 | 7 | 0.923 | 0.962 | 0.942 | 4.08 | 1.53 | 4.18 | 3.97 | 6.64 | 0.345 |

Failure gallery (worst 24 test glyphs): `reports/stage2/gallery/classical_test_01.png`, `reports/stage2/gallery/classical_test_02.png`

