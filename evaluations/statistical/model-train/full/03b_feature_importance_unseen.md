# Permutation Feature Importance, Unseen Set

Metrik, kenaikan RMSE (USD) setelah fitur diacak antar sampel.

Repeats: 10, seed: 42.

## MLP (baseline RMSE = 6.2964 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 9.9987 | 0.1191 | 158.80% |
| IMF_Group1 | 3.7011 | 0.0963 | 58.78% |
| IMF_Group2 | 2.1296 | 0.0429 | 33.82% |
| Residual | -0.2518 | 0.0240 | -4.00% |

## RNN (baseline RMSE = 5.9626 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 11.4174 | 0.1449 | 191.48% |
| IMF_Group1 | 5.0440 | 0.1111 | 84.60% |
| IMF_Group2 | 2.3177 | 0.0687 | 38.87% |
| Residual | -0.0459 | 0.0167 | -0.77% |

## LSTM (baseline RMSE = 6.9208 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 11.0867 | 0.1884 | 160.19% |
| IMF_Group1 | 4.7114 | 0.1135 | 68.08% |
| IMF_Group2 | 1.9011 | 0.0534 | 27.47% |
| Residual | 0.0505 | 0.0598 | 0.73% |

## BiLSTM (baseline RMSE = 6.2894 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 10.6938 | 0.1473 | 170.03% |
| IMF_Group1 | 3.9446 | 0.0880 | 62.72% |
| IMF_Group2 | 2.1781 | 0.0506 | 34.63% |
| Residual | 0.0805 | 0.0381 | 1.28% |

## GRU (baseline RMSE = 6.3614 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 11.0077 | 0.1551 | 173.04% |
| IMF_Group1 | 4.8586 | 0.1180 | 76.38% |
| IMF_Group2 | 2.1616 | 0.0594 | 33.98% |
| Residual | 0.0437 | 0.0328 | 0.69% |

## TCN (baseline RMSE = 6.2872 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 10.4221 | 0.1481 | 165.77% |
| IMF_Group1 | 3.7673 | 0.1142 | 59.92% |
| IMF_Group2 | 2.0037 | 0.0595 | 31.87% |
| Residual | 0.1201 | 0.0541 | 1.91% |

## Transformer (baseline RMSE = 6.3566 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 12.1042 | 0.1926 | 190.42% |
| IMF_Group1 | 5.4185 | 0.1138 | 85.24% |
| IMF_Group2 | 2.5684 | 0.0581 | 40.40% |
| Residual | -0.1849 | 0.0486 | -2.91% |

## Informer (baseline RMSE = 6.9664 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 10.6922 | 0.1730 | 153.48% |
| IMF_Group1 | 4.7838 | 0.1208 | 68.67% |
| IMF_Group2 | 1.5766 | 0.0578 | 22.63% |
| Residual | 0.0703 | 0.0458 | 1.01% |

Semakin tinggi dRMSE atau persen kenaikan, semakin penting fitur tersebut bagi model.
Nilai negatif menandakan pengacakan fitur tidak merugikan (atau justru sedikit membantu, seperti noise) bagi prediksi model tersebut.
