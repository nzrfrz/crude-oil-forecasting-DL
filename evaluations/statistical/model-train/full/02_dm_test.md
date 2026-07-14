# Diebold Mariano Pairwise Test, Test Set

Harvey et al. (1997) small sample correction, h=1, loss adalah squared error.

| Pair (i vs j) | DM Stat | p-value | Sig |
|---|---:|---:|---|
| MLP vs RNN | 3.4711 | 0.0005 | *** |
| MLP vs LSTM | -4.0270 | 0.0001 | *** |
| MLP vs BiLSTM | -0.8114 | 0.4174 | ns |
| MLP vs GRU | -1.2255 | 0.2207 | ns |
| MLP vs TCN | -1.8130 | 0.0701 | ns |
| MLP vs Transformer | -2.8592 | 0.0043 | ** |
| MLP vs Informer | -1.9792 | 0.0481 | * |
| RNN vs MLP | -3.4711 | 0.0005 | *** |
| RNN vs LSTM | -6.8325 | 0.0000 | *** |
| RNN vs BiLSTM | -5.6216 | 0.0000 | *** |
| RNN vs GRU | -7.2131 | 0.0000 | *** |
| RNN vs TCN | -4.5145 | 0.0000 | *** |
| RNN vs Transformer | -6.7224 | 0.0000 | *** |
| RNN vs Informer | -6.6359 | 0.0000 | *** |
| LSTM vs MLP | 4.0270 | 0.0001 | *** |
| LSTM vs RNN | 6.8325 | 0.0000 | *** |
| LSTM vs BiLSTM | 6.8852 | 0.0000 | *** |
| LSTM vs GRU | 6.1893 | 0.0000 | *** |
| LSTM vs TCN | 3.7365 | 0.0002 | *** |
| LSTM vs Transformer | 2.9174 | 0.0036 | ** |
| LSTM vs Informer | 5.7070 | 0.0000 | *** |
| BiLSTM vs MLP | 0.8114 | 0.4174 | ns |
| BiLSTM vs RNN | 5.6216 | 0.0000 | *** |
| BiLSTM vs LSTM | -6.8852 | 0.0000 | *** |
| BiLSTM vs GRU | -1.4868 | 0.1374 | ns |
| BiLSTM vs TCN | -0.7295 | 0.4659 | ns |
| BiLSTM vs Transformer | -4.6129 | 0.0000 | *** |
| BiLSTM vs Informer | -3.5863 | 0.0004 | *** |
| GRU vs MLP | 1.2255 | 0.2207 | ns |
| GRU vs RNN | 7.2131 | 0.0000 | *** |
| GRU vs LSTM | -6.1893 | 0.0000 | *** |
| GRU vs BiLSTM | 1.4868 | 0.1374 | ns |
| GRU vs TCN | -0.2964 | 0.7670 | ns |
| GRU vs Transformer | -3.8697 | 0.0001 | *** |
| GRU vs Informer | -2.9549 | 0.0032 | ** |
| TCN vs MLP | 1.8130 | 0.0701 | ns |
| TCN vs RNN | 4.5145 | 0.0000 | *** |
| TCN vs LSTM | -3.7365 | 0.0002 | *** |
| TCN vs BiLSTM | 0.7295 | 0.4659 | ns |
| TCN vs GRU | 0.2964 | 0.7670 | ns |
| TCN vs Transformer | -2.1079 | 0.0353 | * |
| TCN vs Informer | -0.7914 | 0.4289 | ns |
| Transformer vs MLP | 2.8592 | 0.0043 | ** |
| Transformer vs RNN | 6.7224 | 0.0000 | *** |
| Transformer vs LSTM | -2.9174 | 0.0036 | ** |
| Transformer vs BiLSTM | 4.6129 | 0.0000 | *** |
| Transformer vs GRU | 3.8697 | 0.0001 | *** |
| Transformer vs TCN | 2.1079 | 0.0353 | * |
| Transformer vs Informer | 2.9256 | 0.0035 | ** |
| Informer vs MLP | 1.9792 | 0.0481 | * |
| Informer vs RNN | 6.6359 | 0.0000 | *** |
| Informer vs LSTM | -5.7070 | 0.0000 | *** |
| Informer vs BiLSTM | 3.5863 | 0.0004 | *** |
| Informer vs GRU | 2.9549 | 0.0032 | ** |
| Informer vs TCN | 0.7914 | 0.4289 | ns |
| Informer vs Transformer | -2.9256 | 0.0035 | ** |

Signifikansi:

- *** p<0.001
- ** p<0.01
- * p<0.05
- ns p>=0.05

Catatan, DM<0 berarti model i memiliki squared loss lebih rendah dari model j (model i lebih baik).
