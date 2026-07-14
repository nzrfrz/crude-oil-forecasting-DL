# Permutation Feature Importance, Test Set

Metrik, kenaikan RMSE (USD) setelah fitur diacak antar sampel.

Repeats: 10, seed: 42.

## MLP (baseline RMSE = 5.1681 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 24.8645 | 0.3497 | 481.11% |
| IMF_Group1 | 5.2365 | 0.1119 | 101.32% |
| IMF_Group2 | 2.6428 | 0.0505 | 51.14% |
| Residual | -0.0323 | 0.0166 | -0.63% |

## RNN (baseline RMSE = 4.5414 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 27.2885 | 0.4325 | 600.89% |
| IMF_Group1 | 7.4753 | 0.1065 | 164.61% |
| IMF_Group2 | 3.2377 | 0.0932 | 71.29% |
| Residual | -0.0280 | 0.0209 | -0.62% |

## LSTM (baseline RMSE = 6.3389 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.2044 | 0.4204 | 413.39% |
| IMF_Group1 | 5.9806 | 0.1143 | 94.35% |
| IMF_Group2 | 2.5529 | 0.0557 | 40.27% |
| Residual | -0.1565 | 0.0898 | -2.47% |

## BiLSTM (baseline RMSE = 5.3297 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.3373 | 0.3804 | 494.16% |
| IMF_Group1 | 6.2910 | 0.1233 | 118.04% |
| IMF_Group2 | 2.9232 | 0.0848 | 54.85% |
| Residual | -0.0305 | 0.0847 | -0.57% |

## GRU (baseline RMSE = 5.4059 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.6306 | 0.4196 | 492.62% |
| IMF_Group1 | 7.1222 | 0.1072 | 131.75% |
| IMF_Group2 | 2.6921 | 0.0806 | 49.80% |
| Residual | -0.0337 | 0.0643 | -0.62% |

## TCN (baseline RMSE = 5.4572 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 25.5532 | 0.4073 | 468.25% |
| IMF_Group1 | 6.4748 | 0.3542 | 118.65% |
| IMF_Group2 | 3.0781 | 0.3546 | 56.40% |
| Residual | -0.0041 | 0.1046 | -0.07% |

## Transformer (baseline RMSE = 5.9522 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.5301 | 0.3790 | 445.72% |
| IMF_Group1 | 7.6502 | 0.2053 | 128.53% |
| IMF_Group2 | 3.0676 | 0.1056 | 51.54% |
| Residual | -0.3151 | 0.1007 | -5.29% |

## Informer (baseline RMSE = 5.5975 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.5633 | 0.4365 | 474.56% |
| IMF_Group1 | 7.1759 | 0.1323 | 128.20% |
| IMF_Group2 | 2.4608 | 0.0719 | 43.96% |
| Residual | -0.0286 | 0.0888 | -0.51% |

Semakin tinggi dRMSE atau persen kenaikan, semakin penting fitur tersebut bagi model.
Nilai negatif menandakan pengacakan fitur tidak merugikan (atau justru sedikit membantu, seperti noise) bagi prediksi model tersebut.
