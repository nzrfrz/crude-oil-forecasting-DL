# Diebold Mariano Pairwise Test, Unseen Set

Harvey et al. (1997) small sample correction, h=1, loss adalah squared error.

| Pair (i vs j) | DM Stat | p-value | Sig |
|---|---:|---:|---|
| MLP vs RNN | 2.5397 | 0.0113 | * |
| MLP vs LSTM | -2.5033 | 0.0125 | * |
| MLP vs BiLSTM | 0.0444 | 0.9646 | ns |
| MLP vs GRU | -0.3683 | 0.7127 | ns |
| MLP vs TCN | 0.0686 | 0.9453 | ns |
| MLP vs Transformer | -0.2730 | 0.7849 | ns |
| MLP vs Informer | -2.8391 | 0.0046 | ** |
| RNN vs MLP | -2.5397 | 0.0113 | * |
| RNN vs LSTM | -4.8270 | 0.0000 | *** |
| RNN vs BiLSTM | -4.5838 | 0.0000 | *** |
| RNN vs GRU | -4.2372 | 0.0000 | *** |
| RNN vs TCN | -4.2068 | 0.0000 | *** |
| RNN vs Transformer | -2.6885 | 0.0073 | ** |
| RNN vs Informer | -5.5388 | 0.0000 | *** |
| LSTM vs MLP | 2.5033 | 0.0125 | * |
| LSTM vs RNN | 4.8270 | 0.0000 | *** |
| LSTM vs BiLSTM | 4.7170 | 0.0000 | *** |
| LSTM vs GRU | 4.6503 | 0.0000 | *** |
| LSTM vs TCN | 3.8831 | 0.0001 | *** |
| LSTM vs Transformer | 4.3159 | 0.0000 | *** |
| LSTM vs Informer | -0.4553 | 0.6490 | ns |
| BiLSTM vs MLP | -0.0444 | 0.9646 | ns |
| BiLSTM vs RNN | 4.5838 | 0.0000 | *** |
| BiLSTM vs LSTM | -4.7170 | 0.0000 | *** |
| BiLSTM vs GRU | -1.5565 | 0.1199 | ns |
| BiLSTM vs TCN | 0.0352 | 0.9719 | ns |
| BiLSTM vs Transformer | -0.5990 | 0.5493 | ns |
| BiLSTM vs Informer | -5.2096 | 0.0000 | *** |
| GRU vs MLP | 0.3683 | 0.7127 | ns |
| GRU vs RNN | 4.2372 | 0.0000 | *** |
| GRU vs LSTM | -4.6503 | 0.0000 | *** |
| GRU vs BiLSTM | 1.5565 | 0.1199 | ns |
| GRU vs TCN | 0.8073 | 0.4197 | ns |
| GRU vs Transformer | 0.0628 | 0.9500 | ns |
| GRU vs Informer | -6.3960 | 0.0000 | *** |
| TCN vs MLP | -0.0686 | 0.9453 | ns |
| TCN vs RNN | 4.2068 | 0.0000 | *** |
| TCN vs LSTM | -3.8831 | 0.0001 | *** |
| TCN vs BiLSTM | -0.0352 | 0.9719 | ns |
| TCN vs GRU | -0.8073 | 0.4197 | ns |
| TCN vs Transformer | -0.4751 | 0.6349 | ns |
| TCN vs Informer | -4.0967 | 0.0000 | *** |
| Transformer vs MLP | 0.2730 | 0.7849 | ns |
| Transformer vs RNN | 2.6885 | 0.0073 | ** |
| Transformer vs LSTM | -4.3159 | 0.0000 | *** |
| Transformer vs BiLSTM | 0.5990 | 0.5493 | ns |
| Transformer vs GRU | -0.0628 | 0.9500 | ns |
| Transformer vs TCN | 0.4751 | 0.6349 | ns |
| Transformer vs Informer | -7.3484 | 0.0000 | *** |
| Informer vs MLP | 2.8391 | 0.0046 | ** |
| Informer vs RNN | 5.5388 | 0.0000 | *** |
| Informer vs LSTM | 0.4553 | 0.6490 | ns |
| Informer vs BiLSTM | 5.2096 | 0.0000 | *** |
| Informer vs GRU | 6.3960 | 0.0000 | *** |
| Informer vs TCN | 4.0967 | 0.0000 | *** |
| Informer vs Transformer | 7.3484 | 0.0000 | *** |

Signifikansi:

- *** p<0.001
- ** p<0.01
- * p<0.05
- ns p>=0.05

Catatan, DM<0 berarti model i memiliki squared loss lebih rendah dari model j (model i lebih baik).
