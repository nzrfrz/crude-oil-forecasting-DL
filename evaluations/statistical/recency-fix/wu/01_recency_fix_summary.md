# Recency-Bias Mitigation, Baseline vs Fixed (wu)

Fixed, learned positional encoding plus last timestep pooling.
Baseline, checkpoint Stage 05 asli, sinusoidal PE plus mean pooling, tidak dilatih ulang.

## Transformer

### Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.7047 | 2.9898 | 2.9936 | 2.2688 | 85.78 | 0.3780 | 0.9868 |
| Fixed | 1.5877 | 2.8310 | 2.7994 | 2.1191 | 86.53 | 0.3520 | 0.9885 |

### Timestep Attribution Concentration Score (test set, SHAP)

- Baseline: 0.0340
- Fixed: 0.0965

Skor 0 berarti atribusi menyebar rata ke semua 10 hari lookback, skor 1 berarti terkonsentrasi penuh di satu hari.

Verdict, konsentrasi recency MENINGKAT (0.0340 menjadi 0.0965).

## Informer

### Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.6921 | 2.9908 | 2.9802 | 2.2532 | 84.67 | 0.3752 | 0.9870 |
| Fixed | 1.4374 | 2.5063 | 2.5279 | 2.0324 | 88.81 | 0.3187 | 0.9894 |

### Timestep Attribution Concentration Score (test set, SHAP)

- Baseline: 0.0144
- Fixed: 0.1563

Skor 0 berarti atribusi menyebar rata ke semua 10 hari lookback, skor 1 berarti terkonsentrasi penuh di satu hari.

Verdict, konsentrasi recency MENINGKAT (0.0144 menjadi 0.1563).

