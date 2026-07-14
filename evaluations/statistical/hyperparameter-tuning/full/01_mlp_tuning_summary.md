# Hyperparameter Tuning, MLP (full)

Search: Optuna TPE sampler + MedianPruner, 30 trial.
Search budget: max_epochs=80, patience=12.
Retrain akhir budget: max_epochs=200, patience=20.

Best hyperparameters: `{"h1": 256, "h2": 128, "h3": 32, "dropout": 0.007929413671136584, "lr": 0.0011107496596584855, "weight_decay": 0.0001}`

Best search val loss (MSE): 0.000072

## Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Stage 05) | 3.1983 | 6.9060 | 6.4232 | 5.1681 | 80.48 | 0.4814 | 0.9440 |
| Tuned | 2.6613 | 6.0406 | 5.4014 | 4.6811 | 87.71 | 0.4006 | 0.9540 |

## Unseen Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Stage 05) | 3.1099 | 3.7210 | 3.8767 | 6.2964 | 85.04 | 0.4978 | 0.6527 |
| Tuned | 2.8475 | 3.4375 | 3.4832 | 5.8641 | 88.34 | 0.4558 | 0.6988 |

Catatan, checkpoint tuned disimpan terpisah (`models/full/mlp_tuned_model.pt`), tidak menimpa checkpoint Stage 05. Kalau hasil di atas lebih baik, adopsi sebagai baseline baru adalah keputusan manual terpisah.
