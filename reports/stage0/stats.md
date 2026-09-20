# Stage 0 report

## Pairs

| Font | Pairs | Clean | Flagged |
|---|---|---|---|
| Assurbanipal | 161 | 160 | 1 |
| Santakku | 288 | 267 | 21 |
| **Total** | 449 | 427 | 22 |

## Stroke kinds

| Font | wedge | of which with orphan points | triangle | malformed |
|---|---|---|---|---|
| Assurbanipal | 926 | 0 | 1 | 0 |
| Santakku | 1800 | 35 | 27 | 1 |
| **Total** | 2726 | 35 | 28 | 1 |

## Strokes per glyph

| Strokes | Glyphs |
|---|---|
| 1 | 7 |
| 2 | 24 |
| 3 | 26 |
| 4 | 54 |
| 5 | 93 |
| 6 | 80 |
| 7 | 59 |
| 8 | 39 |
| 9 | 15 |
| 10 | 20 |
| 11 | 16 |
| 12 | 4 |
| 13 | 2 |
| 14 | 4 |
| 15 | 3 |
| 16 | 1 |
| 17 | 2 |

## Split (unit = codepoint, seed = 0, fractions = [0.8, 0.1, 0.1])

| Font | train | val | test |
|---|---|---|---|
| Assurbanipal | 129 | 16 | 16 |
| Santakku | 230 | 29 | 29 |

Codepoints per font-group:

| Group | train | val | test |
|---|---|---|---|
| Assurbanipal | 11 | 1 | 1 |
| Assurbanipal+Santakku | 118 | 15 | 15 |
| Santakku | 112 | 14 | 14 |

Pinned to train: 0x122b9

## Corrections applied

1 glyph(s) read from `data/corrections/` instead of `skeletons/`:

- Santakku 0x12130 HI TIMES BAD

## Flagged glyphs

| Font | Codepoint | Name | Split | Flags |
|---|---|---|---|---|
| Assurbanipal | 0x1207a | DU | train | Stroke 5:triangle |
| Santakku | 0x12029 | ALAN | train | Stroke 1:triangle |
| Santakku | 0x12032 | ANSHE | train | Stroke 3:triangle; Stroke 4:orphans=1; Stroke 5:triangle; Stroke 6:orphans=1 |
| Santakku | 0x1207a | DU | train | Stroke 5:triangle |
| Santakku | 0x12081 | DUG | train | Stroke 5:malformed |
| Santakku | 0x12084 | DUN | train | Stroke 1:orphans=2; Stroke 2:orphans=3; Stroke 3:orphans=2; Stroke 5:orphans=1; Stroke 6:orphans=2; Stroke 7:orphans=2; Stroke 8:orphans=1; Stroke 9:orphans=1; Stroke 10:orphans=2; Stroke 11:orphans=3; Stroke 12:orphans=2 |
| Santakku | 0x120a6 | EZEN TIMES BAD | train | Stroke 1:orphans=1; Stroke 2:orphans=2; Stroke 3:orphans=2; Stroke 4:orphans=2; Stroke 5:orphans=2; Stroke 6:orphans=1; Stroke 7:orphans=2; Stroke 8:orphans=2 |
| Santakku | 0x12118 | GU2 & UN | train | Stroke 1:orphans=1; Stroke 2:orphans=2; Stroke 3:orphans=2; Stroke 4:orphans=1; Stroke 5:orphans=1; Stroke 6:orphans=1 |
| Santakku | 0x1212c | HAL | train | Stroke 1:triangle |
| Santakku | 0x1213e | HUL2 | train | Stroke 6:triangle; Stroke 7:orphans=1 |
| Santakku | 0x1218d | KA2 | train | Stroke 1:orphans=2; Stroke 2:orphans=2; Stroke 3:triangle; Stroke 4:triangle |
| Santakku | 0x121b5 | KUSHU2 | test | Stroke 5:triangle |
| Santakku | 0x121b9 | LAGAB TIMES A | train | Stroke 4:triangle |
| Santakku | 0x12224 | MAH | test | Stroke 4:triangle |
| Santakku | 0x12232 | MUSH | train | Stroke 1:triangle; Stroke 2:triangle; Stroke 3:triangle; Stroke 4:triangle; Stroke 5:triangle; Stroke 6:triangle |
| Santakku | 0x1223f | NA2 | val | Stroke 1:triangle; Stroke 6:orphans=4 |
| Santakku | 0x122b7 | SHA6 (bad!) | train | Stroke 11:orphans=1; Stroke 12:orphans=3 |
| Santakku | 0x122c6 | SHIM | train | Stroke 4:triangle |
| Santakku | 0x122cb | SHIM TIMES GAR | val | Stroke 1:triangle; Stroke 2:orphans=1 |
| Santakku | 0x122dc | SI GUNU | train | Stroke 7:orphans=1 |
| Santakku | 0x12322 | UMBIN | train | Stroke 4:triangle; Stroke 14:triangle |
| Santakku | 0x1238f |  | test | Stroke 1:triangle; Stroke 2:triangle; Stroke 5:triangle; Stroke 6:triangle; Stroke 7:triangle |
