# Stage 1 report: evaluation harness

Match threshold: 10 px on apex distance. Corner error is the mean distance over the 4 wedge points of matched wedges. Exact = no FP, no FN, every matched point within the threshold. Only wedge strokes are evaluated; triangle and malformed strokes are excluded from ground truth.

## Reference predictors

| Predictor | Split | Font | Glyphs | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Corner err (px) | Apex | Head a | Head b | Tail | Exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null | train | all | 359 | 2170 | 0 | 0 | 0 | 2170 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.003 |
| null | train | Assurbanipal | 129 | 733 | 0 | 0 | 0 | 733 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| null | train | Santakku | 230 | 1437 | 0 | 0 | 0 | 1437 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.004 |
| oracle | train | all | 359 | 2170 | 2170 | 2170 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | train | Assurbanipal | 129 | 733 | 733 | 733 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | train | Santakku | 230 | 1437 | 1437 | 1437 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | train | all | 359 | 2170 | 2078 | 1954 | 124 | 216 | 0.940 | 0.900 | 0.920 | 2.53 | 2.49 | 2.53 | 2.51 | 2.59 | 0.443 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | train | Assurbanipal | 129 | 733 | 709 | 661 | 48 | 72 | 0.932 | 0.902 | 0.917 | 2.51 | 2.48 | 2.47 | 2.53 | 2.57 | 0.473 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | train | Santakku | 230 | 1437 | 1369 | 1293 | 76 | 144 | 0.944 | 0.900 | 0.922 | 2.54 | 2.50 | 2.56 | 2.51 | 2.60 | 0.426 |
| null | val | all | 45 | 256 | 0 | 0 | 0 | 256 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| null | val | Assurbanipal | 16 | 79 | 0 | 0 | 0 | 79 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| null | val | Santakku | 29 | 177 | 0 | 0 | 0 | 177 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| oracle | val | all | 45 | 256 | 256 | 256 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | val | Assurbanipal | 16 | 79 | 79 | 79 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | val | Santakku | 29 | 177 | 177 | 177 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | val | all | 45 | 256 | 248 | 229 | 19 | 27 | 0.923 | 0.895 | 0.909 | 2.50 | 2.41 | 2.54 | 2.62 | 2.45 | 0.467 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | val | Assurbanipal | 16 | 79 | 76 | 73 | 3 | 6 | 0.961 | 0.924 | 0.942 | 2.54 | 2.21 | 2.51 | 2.77 | 2.66 | 0.438 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | val | Santakku | 29 | 177 | 172 | 156 | 16 | 21 | 0.907 | 0.881 | 0.894 | 2.49 | 2.50 | 2.55 | 2.55 | 2.36 | 0.483 |
| null | test | all | 45 | 300 | 0 | 0 | 0 | 300 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| null | test | Assurbanipal | 16 | 114 | 0 | 0 | 0 | 114 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| null | test | Santakku | 29 | 186 | 0 | 0 | 0 | 186 | 0.000 | 0.000 | 0.000 | nan | nan | nan | nan | nan | 0.000 |
| oracle | test | all | 45 | 300 | 300 | 300 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | test | Assurbanipal | 16 | 114 | 114 | 114 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| oracle | test | Santakku | 29 | 186 | 186 | 186 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.000 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | test | all | 45 | 300 | 287 | 265 | 22 | 35 | 0.923 | 0.883 | 0.903 | 2.49 | 2.47 | 2.51 | 2.57 | 2.41 | 0.333 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | test | Assurbanipal | 16 | 114 | 110 | 106 | 4 | 8 | 0.964 | 0.930 | 0.946 | 2.46 | 2.27 | 2.50 | 2.69 | 2.38 | 0.375 |
| jittered_oracle(sigma=2.0,drop=0.1,spurious=0.05) | test | Santakku | 29 | 186 | 177 | 159 | 18 | 27 | 0.898 | 0.855 | 0.876 | 2.51 | 2.60 | 2.51 | 2.49 | 2.44 | 0.310 |

Reading the table: `null` is the floor (recall 0). `oracle` is the ceiling and must show precision, recall, F1 and exact all equal to 1 with zero error. `jittered_oracle` perturbs the ground truth by 2 px Gaussian noise per point, drops 10% of wedges and adds 5% spurious ones; it shows what the numbers look like for a model that is nearly right, and its gallery below demonstrates the failure renderer.

