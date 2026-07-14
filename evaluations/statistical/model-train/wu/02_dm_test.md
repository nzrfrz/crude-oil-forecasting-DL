# Diebold Mariano Pairwise Test, Test Set

Harvey et al. (1997) small sample correction, h=1, loss adalah squared error.

| Pair (i vs j) | DM Stat | p-value | Sig |
|---|---:|---:|---|
| MLP vs RNN | 19.1659 | 0.0000 | *** |
| MLP vs LSTM | 14.3899 | 0.0000 | *** |
| MLP vs BiLSTM | 17.0446 | 0.0000 | *** |
| MLP vs GRU | 15.0734 | 0.0000 | *** |
| MLP vs TCN | 19.6528 | 0.0000 | *** |
| MLP vs Transformer | 11.5630 | 0.0000 | *** |
| MLP vs Informer | 12.4147 | 0.0000 | *** |
| RNN vs MLP | -19.1659 | 0.0000 | *** |
| RNN vs LSTM | -5.5472 | 0.0000 | *** |
| RNN vs BiLSTM | -2.5202 | 0.0118 | * |
| RNN vs GRU | -4.7426 | 0.0000 | *** |
| RNN vs TCN | 0.7808 | 0.4350 | ns |
| RNN vs Transformer | -7.3741 | 0.0000 | *** |
| RNN vs Informer | -8.1535 | 0.0000 | *** |
| LSTM vs MLP | -14.3899 | 0.0000 | *** |
| LSTM vs RNN | 5.5472 | 0.0000 | *** |
| LSTM vs BiLSTM | 6.5420 | 0.0000 | *** |
| LSTM vs GRU | 1.0494 | 0.2941 | ns |
| LSTM vs TCN | 5.1661 | 0.0000 | *** |
| LSTM vs Transformer | -5.6426 | 0.0000 | *** |
| LSTM vs Informer | -5.6375 | 0.0000 | *** |
| BiLSTM vs MLP | -17.0446 | 0.0000 | *** |
| BiLSTM vs RNN | 2.5202 | 0.0118 | * |
| BiLSTM vs LSTM | -6.5420 | 0.0000 | *** |
| BiLSTM vs GRU | -3.9234 | 0.0001 | *** |
| BiLSTM vs TCN | 2.6738 | 0.0076 | ** |
| BiLSTM vs Transformer | -7.3248 | 0.0000 | *** |
| BiLSTM vs Informer | -8.2541 | 0.0000 | *** |
| GRU vs MLP | -15.0734 | 0.0000 | *** |
| GRU vs RNN | 4.7426 | 0.0000 | *** |
| GRU vs LSTM | -1.0494 | 0.2941 | ns |
| GRU vs BiLSTM | 3.9234 | 0.0001 | *** |
| GRU vs TCN | 4.3098 | 0.0000 | *** |
| GRU vs Transformer | -6.2938 | 0.0000 | *** |
| GRU vs Informer | -9.0292 | 0.0000 | *** |
| TCN vs MLP | -19.6528 | 0.0000 | *** |
| TCN vs RNN | -0.7808 | 0.4350 | ns |
| TCN vs LSTM | -5.1661 | 0.0000 | *** |
| TCN vs BiLSTM | -2.6738 | 0.0076 | ** |
| TCN vs GRU | -4.3098 | 0.0000 | *** |
| TCN vs Transformer | -7.3307 | 0.0000 | *** |
| TCN vs Informer | -7.5601 | 0.0000 | *** |
| Transformer vs MLP | -11.5630 | 0.0000 | *** |
| Transformer vs RNN | 7.3741 | 0.0000 | *** |
| Transformer vs LSTM | 5.6426 | 0.0000 | *** |
| Transformer vs BiLSTM | 7.3248 | 0.0000 | *** |
| Transformer vs GRU | 6.2938 | 0.0000 | *** |
| Transformer vs TCN | 7.3307 | 0.0000 | *** |
| Transformer vs Informer | 0.5351 | 0.5926 | ns |
| Informer vs MLP | -12.4147 | 0.0000 | *** |
| Informer vs RNN | 8.1535 | 0.0000 | *** |
| Informer vs LSTM | 5.6375 | 0.0000 | *** |
| Informer vs BiLSTM | 8.2541 | 0.0000 | *** |
| Informer vs GRU | 9.0292 | 0.0000 | *** |
| Informer vs TCN | 7.5601 | 0.0000 | *** |
| Informer vs Transformer | -0.5351 | 0.5926 | ns |

Signifikansi:

- *** p<0.001
- ** p<0.01
- * p<0.05
- ns p>=0.05

Catatan, DM<0 berarti model i memiliki squared loss lebih rendah dari model j (model i lebih baik).
