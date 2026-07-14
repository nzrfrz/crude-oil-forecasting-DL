# Recency-Bias Mitigation, Baseline vs Fixed (full)

Fixed, learned positional encoding plus last timestep pooling.
Baseline, checkpoint Stage 05 asli, sinusoidal PE plus mean pooling, tidak dilatih ulang.

## Transformer

### Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.5398 | 6.4038 | 6.1244 | 5.9522 | 82.33 | 0.5328 | 0.9257 |
| Fixed | 3.5732 | 6.9463 | 6.4631 | 6.1536 | 82.64 | 0.5378 | 0.9205 |

### Unseen Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.3258 | 4.0147 | 4.0197 | 6.3566 | 83.08 | 0.5323 | 0.6461 |
| Fixed | 3.3878 | 4.0824 | 4.0875 | 6.7120 | 82.35 | 0.5423 | 0.6054 |

### Timestep Attribution Concentration Score (test set, SHAP)

- Baseline: 0.0765
- Fixed: 0.1625

Skor 0 berarti atribusi menyebar rata ke semua 10 hari lookback, skor 1 berarti terkonsentrasi penuh di satu hari.

Verdict, konsentrasi recency MENINGKAT (0.0765 menjadi 0.1625).

## Informer

### Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.4012 | 6.7074 | 6.5110 | 5.5975 | 81.51 | 0.5119 | 0.9343 |
| Fixed | 2.9913 | 5.9306 | 5.8595 | 4.6434 | 85.95 | 0.4502 | 0.9548 |

### Unseen Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.5364 | 4.2453 | 4.2573 | 6.9664 | 80.39 | 0.5661 | 0.5749 |
| Fixed | 3.3433 | 4.0490 | 4.0861 | 6.2997 | 83.49 | 0.5352 | 0.6524 |

### Timestep Attribution Concentration Score (test set, SHAP)

- Baseline: 0.0743
- Fixed: 0.1590

Skor 0 berarti atribusi menyebar rata ke semua 10 hari lookback, skor 1 berarti terkonsentrasi penuh di satu hari.

Verdict, konsentrasi recency MENINGKAT (0.0743 menjadi 0.1590).

