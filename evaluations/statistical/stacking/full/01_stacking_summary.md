# Stacking Ensemble, Exhaustive Combination Search (full)

Validation windows, 774. Test windows, 968.
Unseen windows, 969.

## Seleksi Varian Base Learner (MAE Test)

- MLP, Tuned(MAE=2.6613), Baseline(MAE=3.1983), terpilih Tuned
- RNN, Baseline(MAE=2.7962), terpilih Baseline
- LSTM, Baseline(MAE=3.4650), terpilih Baseline
- BiLSTM, Baseline(MAE=3.0547), terpilih Baseline
- GRU, Baseline(MAE=3.1805), terpilih Baseline
- TCN, Baseline(MAE=3.1283), terpilih Baseline
- Transformer, Baseline(MAE=3.5398), Fixed(MAE=3.5732), terpilih Baseline
- Informer, Fixed(MAE=2.9913), Baseline(MAE=3.4012), terpilih Fixed

## Ringkasan Pencarian Kombinasi

Total kombinasi dievaluasi, 364 (ukuran 3 sampai 5 dari 8 base learner x 2 meta-learner). Kriteria rangking, MAE validation set.

## Top 10 Kombinasi (MAE Validation)

| Rank | Anggota | Meta-Learner | MAE Validation |
|---:|---|---|---:|
| 1 | MLP, BiLSTM, GRU, TCN, Transformer | RidgeCV | 0.8534 |
| 2 | MLP, BiLSTM, TCN, Transformer | RidgeCV | 0.8536 |
| 3 | MLP, BiLSTM, TCN, Transformer, Informer | RidgeCV | 0.8536 |
| 4 | MLP, RNN, BiLSTM, TCN, Transformer | RidgeCV | 0.8541 |
| 5 | MLP, BiLSTM, GRU, TCN, Informer | RidgeCV | 0.8544 |
| 6 | MLP, GRU, TCN, Transformer, Informer | RidgeCV | 0.8548 |
| 7 | MLP, LSTM, BiLSTM, TCN, Transformer | RidgeCV | 0.8549 |
| 8 | MLP, GRU, TCN, Informer | RidgeCV | 0.8549 |
| 9 | MLP, LSTM, GRU, TCN, Transformer | RidgeCV | 0.8553 |
| 10 | MLP, BiLSTM, GRU, TCN | RidgeCV | 0.8557 |

## Kombinasi Pemenang

Anggota, MLP, BiLSTM, GRU, TCN, Transformer.
Meta-learner, RidgeCV.
MAE validation, 0.8534.

Alpha terpilih, 100.0000.
Intercept, 0.2737.

Bobot per anggota,
- MLP, 0.3285
- BiLSTM, 0.2180
- GRU, 0.2036
- TCN, 0.2850
- Transformer, -0.0431

## Tabel Metrik, Skala Asli USD

### Test Set

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 2.6613 | 6.0406 | 5.4014 | 4.6811 | 87.71 | 0.4006 | 0.9540 |
| RNN | 2.7962 | 5.5286 | 5.5493 | 4.5414 | 86.78 | 0.4209 | 0.9567 |
| LSTM | 3.4650 | 6.8126 | 6.3931 | 6.3389 | 84.30 | 0.5215 | 0.9157 |
| BiLSTM | 3.0547 | 6.0119 | 5.8313 | 5.3297 | 85.95 | 0.4598 | 0.9404 |
| GRU | 3.1805 | 6.3740 | 6.3115 | 5.4059 | 84.71 | 0.4787 | 0.9387 |
| TCN | 3.1283 | 6.5207 | 6.0377 | 5.4572 | 84.40 | 0.4709 | 0.9375 |
| Transformer | 3.5398 | 6.4038 | 6.1244 | 5.9522 | 82.33 | 0.5328 | 0.9257 |
| Informer | 2.9913 | 5.9306 | 5.8595 | 4.6434 | 85.95 | 0.4502 | 0.9548 |
| Stack(MLP+BiLSTM+GRU+TCN+Transformer)-RidgeCV | 2.7400 | 5.7508 | 5.3822 | 4.8345 | 86.36 | 0.4124 | 0.9510 |

### Unseen Set

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 2.8475 | 3.4375 | 3.4832 | 5.8641 | 88.34 | 0.4558 | 0.6988 |
| RNN | 2.9908 | 3.6025 | 3.6411 | 5.9626 | 87.00 | 0.4787 | 0.6886 |
| LSTM | 3.2860 | 3.9014 | 3.8877 | 6.9208 | 86.38 | 0.5260 | 0.5805 |
| BiLSTM | 3.0320 | 3.6093 | 3.6477 | 6.2894 | 88.13 | 0.4853 | 0.6535 |
| GRU | 3.0896 | 3.6906 | 3.7163 | 6.3614 | 86.38 | 0.4945 | 0.6455 |
| TCN | 3.0933 | 3.7214 | 3.7641 | 6.2872 | 88.44 | 0.4951 | 0.6538 |
| Transformer | 3.3258 | 4.0147 | 4.0197 | 6.3566 | 83.08 | 0.5323 | 0.6461 |
| Informer | 3.3433 | 4.0490 | 4.0861 | 6.2997 | 83.49 | 0.5352 | 0.6524 |
| Stack(MLP+BiLSTM+GRU+TCN+Transformer)-RidgeCV | 2.8759 | 3.4266 | 3.4850 | 6.0805 | 89.47 | 0.4603 | 0.6761 |

## Diebold-Mariano Test, Kombinasi Pemenang vs Model Individu Terbaik

Model individu terbaik (MAE test terendah di antara 8 base learner), MLP.

Test set, DM statistic 1.9018, p-value 0.0575 (ns).
Unseen set, DM statistic 4.6783, p-value 0.0000 (***).

Statistic negatif berarti kombinasi pemenang punya loss lebih rendah dari model individu terbaik, positif berarti sebaliknya. Signifikan (p < 0.05) berarti perbedaan performa bukan kebetulan statistik.

## Catatan Kejujuran dan Batasan

Rangking kombinasi berbasis MAE validation set, bukan test set, untuk mencegah kebocoran ke test set, tapi validation set relatif kecil, ada risiko overfitting seleksi, kombinasi menang di validation belum tentu menang telak di test. Tabel Top 10 di atas disertakan supaya pembaca bisa menilai seberapa dekat marjin kemenangan kombinasi pemenang dibanding kombinasi lain. Hasil dilaporkan apa adanya, termasuk kalau kombinasi pemenang ternyata tidak mengalahkan model individu terbaik secara signifikan.
