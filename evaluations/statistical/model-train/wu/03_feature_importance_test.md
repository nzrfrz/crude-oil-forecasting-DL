# Permutation Feature Importance, Test Set

Metrik, kenaikan RMSE (USD) setelah fitur diacak antar sampel.

Repeats: 10, seed: 42.

## MLP (baseline RMSE = 3.2932 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 26.2991 | 0.2054 | 798.59% |
| IMF_Group1 | 3.7558 | 0.0977 | 114.05% |
| IMF_Group2 | 1.1934 | 0.0299 | 36.24% |
| Residual | 0.0627 | 0.0197 | 1.90% |

## RNN (baseline RMSE = 1.8954 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 27.7756 | 0.2120 | 1465.39% |
| IMF_Group1 | 5.1419 | 0.1145 | 271.28% |
| IMF_Group2 | 2.4984 | 0.0336 | 131.81% |
| Residual | 0.0081 | 0.0063 | 0.43% |

## LSTM (baseline RMSE = 2.0853 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 28.1193 | 0.2111 | 1348.43% |
| IMF_Group1 | 5.6174 | 0.1288 | 269.37% |
| IMF_Group2 | 2.1829 | 0.0312 | 104.68% |
| Residual | 0.2060 | 0.0178 | 9.88% |

## BiLSTM (baseline RMSE = 1.9594 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 27.6273 | 0.2162 | 1410.01% |
| IMF_Group1 | 5.0442 | 0.1182 | 257.44% |
| IMF_Group2 | 2.2928 | 0.0392 | 117.02% |
| Residual | 0.1036 | 0.0134 | 5.29% |

## GRU (baseline RMSE = 2.0638 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 28.0888 | 0.2020 | 1361.05% |
| IMF_Group1 | 5.5139 | 0.1274 | 267.18% |
| IMF_Group2 | 2.0319 | 0.0293 | 98.46% |
| Residual | 0.1149 | 0.0142 | 5.57% |

## TCN (baseline RMSE = 1.8762 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 27.7171 | 0.2181 | 1477.34% |
| IMF_Group1 | 5.2374 | 0.1184 | 279.15% |
| IMF_Group2 | 2.4432 | 0.0459 | 130.23% |
| Residual | 0.1875 | 0.0143 | 9.99% |

## Transformer (baseline RMSE = 2.2688 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 28.9256 | 0.2225 | 1274.93% |
| IMF_Group1 | 5.9659 | 0.1415 | 262.96% |
| IMF_Group2 | 2.0752 | 0.0357 | 91.47% |
| Residual | 0.7635 | 0.0335 | 33.65% |

## Informer (baseline RMSE = 2.2532 USD)

| Feature | dRMSE (USD) | Std | % Increase |
|---|---:|---:|---:|
| Trend | 28.1972 | 0.1947 | 1251.40% |
| IMF_Group1 | 5.6220 | 0.1373 | 249.51% |
| IMF_Group2 | 1.8270 | 0.0311 | 81.08% |
| Residual | 0.3393 | 0.0220 | 15.06% |

Semakin tinggi dRMSE atau persen kenaikan, semakin penting fitur tersebut bagi model.
Nilai negatif menandakan pengacakan fitur tidak merugikan (atau justru sedikit membantu, seperti noise) bagi prediksi model tersebut.
