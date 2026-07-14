# Hyperparameter Tuning, MLP (wu)

Search: Optuna TPE sampler + MedianPruner, 30 trial.
Search budget: max_epochs=80, patience=12.
Retrain akhir budget: max_epochs=200, patience=20.

Best hyperparameters: `{"h1": 256, "h2": 128, "h3": 32, "dropout": 0.03625603004646558, "lr": 0.0016522729299476796, "weight_decay": 0.0001}`

Best search val loss (MSE): 0.000195

## Test Set Metrics

| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Stage 05) | 2.4679 | 4.3933 | 4.4937 | 3.2932 | 81.89 | 0.5472 | 0.9722 |
| Tuned | 1.2938 | 2.2953 | 2.2922 | 1.9330 | 88.81 | 0.2868 | 0.9904 |

Catatan, checkpoint tuned disimpan terpisah (`models/wu/mlp_tuned_model.pt`), tidak menimpa checkpoint Stage 05. Kalau hasil di atas lebih baik, adopsi sebagai baseline baru adalah keputusan manual terpisah.
