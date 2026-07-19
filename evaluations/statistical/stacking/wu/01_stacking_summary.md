# Stacking Ensemble, Exhaustive Combination Search (wu)

Validation windows, 646. Test windows, 1618.

## Seleksi Varian Base Learner (MAE Test)

- MLP, Tuned(MAE=1.2938), Baseline(MAE=2.4679), terpilih Tuned
- RNN, Baseline(MAE=1.2945), terpilih Baseline
- LSTM, Baseline(MAE=1.5556), terpilih Baseline
- BiLSTM, Baseline(MAE=1.3788), terpilih Baseline
- GRU, Baseline(MAE=1.4934), terpilih Baseline
- TCN, Baseline(MAE=1.2462), terpilih Baseline
- Transformer, Fixed(MAE=1.5877), Baseline(MAE=1.7047), terpilih Fixed
- Informer, Fixed(MAE=1.4374), Baseline(MAE=1.6921), terpilih Fixed

## Ringkasan Pencarian Kombinasi

Total kombinasi dievaluasi, 364 (ukuran 3 sampai 5 dari 8 base learner x 2 meta-learner). Kriteria rangking, MAE validation set.

## Top 10 Kombinasi (MAE Validation)

| Rank | Anggota | Meta-Learner | MAE Validation |
|---:|---|---|---:|
| 1 | MLP, RNN, BiLSTM, GRU, Informer | RidgeCV | 1.3580 |
| 2 | MLP, RNN, LSTM, GRU, Informer | RidgeCV | 1.3619 |
| 3 | MLP, RNN, BiLSTM, Transformer, Informer | RidgeCV | 1.3639 |
| 4 | MLP, RNN, LSTM, BiLSTM, Informer | RidgeCV | 1.3650 |
| 5 | MLP, RNN, BiLSTM, TCN, Informer | RidgeCV | 1.3651 |
| 6 | MLP, RNN, BiLSTM, Informer | RidgeCV | 1.3654 |
| 7 | MLP, RNN, BiLSTM, GRU, Transformer | RidgeCV | 1.3673 |
| 8 | MLP, RNN, LSTM, GRU, Transformer | RidgeCV | 1.3685 |
| 9 | MLP, RNN, LSTM, Transformer, Informer | RidgeCV | 1.3696 |
| 10 | MLP, RNN, GRU, Transformer, Informer | RidgeCV | 1.3696 |

## Kombinasi Pemenang

Anggota, MLP, RNN, BiLSTM, GRU, Informer.
Meta-learner, RidgeCV.
MAE validation, 1.3580.

Alpha terpilih, 50.0000.
Intercept, 2.7150.

Bobot per anggota,
- MLP, 0.3694
- RNN, 0.6947
- BiLSTM, 0.3129
- GRU, -0.1761
- Informer, -0.2308

## Tabel Metrik, Skala Asli USD

### Test Set

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1.2938 | 2.2953 | 2.2922 | 1.9330 | 88.81 | 0.2868 | 0.9904 |
| RNN | 1.2945 | 2.3309 | 2.3301 | 1.8954 | 88.57 | 0.2870 | 0.9908 |
| LSTM | 1.5556 | 2.7669 | 2.7419 | 2.0853 | 86.40 | 0.3449 | 0.9889 |
| BiLSTM | 1.3788 | 2.4473 | 2.4337 | 1.9594 | 87.70 | 0.3057 | 0.9902 |
| GRU | 1.4934 | 2.6634 | 2.6451 | 2.0638 | 86.46 | 0.3311 | 0.9891 |
| TCN | 1.2462 | 2.2432 | 2.2397 | 1.8762 | 89.18 | 0.2763 | 0.9910 |
| Transformer | 1.5877 | 2.8310 | 2.7994 | 2.1191 | 86.53 | 0.3520 | 0.9885 |
| Informer | 1.4374 | 2.5063 | 2.5279 | 2.0324 | 88.81 | 0.3187 | 0.9894 |
| Stack(MLP+RNN+BiLSTM+GRU+Informer)-RidgeCV | 1.8996 | 3.5893 | 3.5143 | 2.3829 | 83.75 | 0.4212 | 0.9855 |

## Diebold-Mariano Test, Kombinasi Pemenang vs Model Individu Terbaik

Model individu terbaik (MAE test terendah di antara 8 base learner), TCN.

Test set, DM statistic 15.9119, p-value 0.0000 (***).

Statistic negatif berarti kombinasi pemenang punya loss lebih rendah dari model individu terbaik, positif berarti sebaliknya. Signifikan (p < 0.05) berarti perbedaan performa bukan kebetulan statistik.

## Catatan Kejujuran dan Batasan

Rangking kombinasi berbasis MAE validation set, bukan test set, untuk mencegah kebocoran ke test set, tapi validation set relatif kecil, ada risiko overfitting seleksi, kombinasi menang di validation belum tentu menang telak di test. Tabel Top 10 di atas disertakan supaya pembaca bisa menilai seberapa dekat marjin kemenangan kombinasi pemenang dibanding kombinasi lain. Hasil dilaporkan apa adanya, termasuk kalau kombinasi pemenang ternyata tidak mengalahkan model individu terbaik secara signifikan.
