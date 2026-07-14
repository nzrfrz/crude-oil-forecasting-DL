# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
# STAGE 11: HYPERPARAMETER TUNING (MLP)
# WTI Crude Oil Price Forecasting, CEEMDAN 4-component input (Final Task DL)
#
# Stage 09 XAI showed MLP as the weakest model on the full-run test set
# (MAPE 6.91%, worst of all 8) and weakest on the wu-run test set (MAPE
# 4.39%) -- unlike Transformer/Informer, MLP's weakness is not a structural
# attention/pooling problem (it has no lookback window at all, so there is
# nothing for XAI to diagnose as "unfocused"), it is a plain feedforward
# capacity/regularization problem, exactly the kind Optuna search is built
# to fix. In the reference project's Stage 06 tuning, the analogous search
# cut MLP's test MAE by ~51% while it made Transformer/Informer worse (see
# README-06-tuning-findings.md) -- so tuning is targeted at MLP only here;
# Transformer/Informer's diagnosed problem is structural and addressed in
# 10-recency-bias-fix.py instead, not by hyperparameter search.
#
# This script does NOT touch 05-dl-model-training.py or its checkpoint. The
# tuned model is trained and saved separately (models/{tag}/mlp_tuned_model.pt)
# so Stage 05's results stay reproducible; adopting the tuned weights as the
# new MLP baseline is a manual follow-up decision, not automatic here.
#
# Adaptasi dari D:\Coding\#bigdata\crude-oil-forecasting-DL\06-hyperparameter-tuning.py,
# dipersempit ke MLP saja (Transformer/Informer sengaja tidak dituning generik,
# lihat alasan di atas), N_FEATURES=4, data dari dataset/splits/<tag>/splits.npz.
# ==================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Tee:
  """Mirrors every write to multiple streams (console + log file)."""

  def __init__(self, *streams):
    self.streams = streams

  def write(self, data):
    for s in self.streams:
      s.write(data)
      s.flush()

  def flush(self):
    for s in self.streams:
      s.flush()


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("--run", choices=["full", "wu"], required=True)
ap.add_argument("--n-trials", type=int, default=30)
ARGS = ap.parse_args()
TAG = ARGS.run
N_TRIALS = ARGS.n_trials

SEED = 42
LOOKBACK = 10
N_FEATURES = 4
BATCH_SIZE = 32
VAL_FRACTION = 0.10
DPI = 300

MAX_EPOCHS = 200          # final retrain budget (mirrors Stage 05)
PATIENCE = 20
TUNE_MAX_EPOCHS = 80      # search budget (speed over precision)
TUNE_PATIENCE = 12

LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-5

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
MODEL_DIR = f"models/{TAG}"
STAT_DIR = f"evaluations/statistical/hyperparameter-tuning/{TAG}"
GRAPH_DIR = f"evaluations/graphical/hyperparameter-tuning/{TAG}"

BASELINE_CHECKPOINT = f"{MODEL_DIR}/mlp_model.pt"
TUNED_CHECKPOINT = f"{MODEL_DIR}/mlp_tuned_model.pt"


# ============================================================
# SECTION 2: DATA LOADING (pre-windowed splits from 04-fe-and-split.py)
# ============================================================

def load_data():
  d = np.load(f"{DATASET_DIR}/splits.npz", allow_pickle=True)
  scaler_X = joblib.load(f"{SCALER_DIR}/scaler_X.pkl")
  scaler_y = joblib.load(f"{SCALER_DIR}/scaler_y.pkl")
  has_unseen = "X_unseen" in d.files

  splits = {
      "train": (d["X_train"].astype(np.float32), d["y_train"].astype(np.float32)),
      "test":  (d["X_test"].astype(np.float32),  d["y_test"].astype(np.float32)),
  }
  if has_unseen:
    splits["unseen"] = (d["X_unseen"].astype(np.float32), d["y_unseen"].astype(np.float32))
  return splits, scaler_X, scaler_y


def make_loader(X, y, batch_size, shuffle=False):
  X_t = torch.tensor(X, dtype=torch.float32)
  y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
  return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


# ============================================================
# SECTION 3: MODEL DEFINITIONS
# MLPModelBaseline is an exact copy of Stage 05's MLPModel (fixed widths),
# used only to load the Stage 05 checkpoint for "before" comparison.
# MLPModelTunable takes widths/dropout as constructor args for the search.
# ============================================================

class MLPModelBaseline(nn.Module):
  """Exact copy of Stage 05's MLPModel. Only used to load the existing
  Stage 05 checkpoint."""

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(N_FEATURES, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32),          nn.ReLU(),
        nn.Linear(32, 1)
    )

  def forward(self, x):
    return self.net(x)


class MLPModelTunable(nn.Module):
  """Same 4-layer MLP shape as Stage 05, with widths/dropout exposed for
  Optuna to search."""

  def __init__(self, h1=128, h2=64, h3=32, dropout=0.2):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(N_FEATURES, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2),         nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, h3),         nn.ReLU(),
        nn.Linear(h3, 1)
    )

  def forward(self, x):
    return self.net(x)


# ============================================================
# SECTION 4: TRAINING INFRASTRUCTURE (mirrors Stage 05, plus Optuna pruning)
# ============================================================

class EarlyStopping:
  def __init__(self, patience=20, min_delta=1e-6):
    self.patience = patience
    self.min_delta = min_delta
    self.best_loss = np.inf
    self.counter = 0
    self.best_state = None

  def step(self, val_loss, model):
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
      self.counter += 1
    return self.counter >= self.patience

  def restore_best(self, model):
    if self.best_state:
      model.load_state_dict(self.best_state)


def train_model(model, train_loader, val_loader, lr, device, weight_decay=0.0,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, trial=None):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
  )
  criterion = nn.MSELoss()
  stopper = EarlyStopping(patience=patience)
  history = {'train_loss': [], 'val_loss': []}

  for epoch in range(1, max_epochs + 1):
    model.train()
    tr_loss = 0.0
    for X_b, y_b in train_loader:
      X_b, y_b = X_b.to(device), y_b.to(device)
      optimizer.zero_grad()
      loss = criterion(model(X_b), y_b)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      tr_loss += loss.item() * len(X_b)
    tr_loss /= len(train_loader.dataset)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
      for X_b, y_b in val_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        va_loss += criterion(model(X_b), y_b).item() * len(X_b)
    va_loss /= len(val_loader.dataset)

    scheduler.step(va_loss)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)

    if trial is not None:
      trial.report(va_loss, epoch)
      if trial.should_prune():
        raise optuna.TrialPruned()

    if stopper.step(va_loss, model):
      break

  stopper.restore_best(model)
  return model, history


# ============================================================
# SECTION 5: EVALUATION UTILITIES (mirrors Stage 05's compute_metrics)
# ============================================================

def predict_array(model, X_arr, device, batch_size=64):
  model.eval()
  X_t = torch.tensor(X_arr, dtype=torch.float32)
  preds = []
  with torch.no_grad():
    for i in range(0, len(X_t), batch_size):
      preds.append(model(X_t[i: i + batch_size].to(device)).cpu().numpy())
  return np.concatenate(preds).flatten()


def compute_metrics(y_true, y_pred, prev_price):
  mae = float(np.mean(np.abs(y_true - y_pred)))
  rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
  mask = np.abs(y_true) > 1e-8
  mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
  smape_denom = np.abs(y_true) + np.abs(y_pred)
  smape_mask = smape_denom > 1e-8
  smape = float(np.mean(
      2.0 * np.abs(y_true[smape_mask] - y_pred[smape_mask]) / smape_denom[smape_mask]
  ) * 100.0)
  true_dir = np.sign(y_true - prev_price)
  pred_dir = np.sign(y_pred - prev_price)
  da = float(np.mean(true_dir == pred_dir) * 100.0)
  naive_mae = float(np.mean(np.abs(y_true - prev_price)))
  mase = float(mae / naive_mae) if naive_mae > 1e-8 else float('nan')
  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
  r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else float('nan')
  return {'MAE': mae, 'MAPE': mape, 'SMAPE': smape, 'RMSE': rmse, 'DA': da,
          'MASE': mase, 'R2': r2}


def prev_price_inv(X_last_timestep, scaler_X):
  dummy = np.zeros((len(X_last_timestep), N_FEATURES), dtype=np.float32)
  dummy[:, 0] = X_last_timestep[:, 0]
  return scaler_X.inverse_transform(dummy)[:, 0]


# ============================================================
# SECTION 6: OPTUNA OBJECTIVE
# ============================================================

def objective_mlp(trial, train_ld, val_ld, device):
  h1 = trial.suggest_categorical('h1', [64, 128, 256])
  h2 = trial.suggest_categorical('h2', [32, 64, 128])
  h3 = trial.suggest_categorical('h3', [16, 32, 64])
  dropout = trial.suggest_float('dropout', 0.0, 0.4)
  lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
  weight_decay = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4])

  model = MLPModelTunable(h1, h2, h3, dropout)
  _, hist = train_model(
      model, train_ld, val_ld, lr, device, weight_decay=weight_decay,
      max_epochs=TUNE_MAX_EPOCHS, patience=TUNE_PATIENCE, trial=trial
  )
  return min(hist['val_loss'])


# ============================================================
# SECTION 7: REPORTS
# ============================================================

def save_tuning_report(best_params, best_val_loss, metrics_base, metrics_tuned,
                       split_keys, filename):
  lines = [f"# Hyperparameter Tuning, MLP ({TAG})", "",
           f"Search: Optuna TPE sampler + MedianPruner, {N_TRIALS} trial.",
           f"Search budget: max_epochs={TUNE_MAX_EPOCHS}, patience={TUNE_PATIENCE}.",
           f"Retrain akhir budget: max_epochs={MAX_EPOCHS}, patience={PATIENCE}.",
           "",
           f"Best hyperparameters: `{json.dumps(best_params)}`", "",
           f"Best search val loss (MSE): {best_val_loss:.6f}", ""]

  for sk in split_keys:
    lines.append(f"## {sk.capitalize()} Set Metrics")
    lines.append("")
    lines.append("| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, m in [("Baseline (Stage 05)", metrics_base[sk]), ("Tuned", metrics_tuned[sk])]:
      lines.append(
          f"| {label} | {m['MAE']:.4f} | {m['MAPE']:.4f} | {m['SMAPE']:.4f} "
          f"| {m['RMSE']:.4f} | {m['DA']:.2f} | {m['MASE']:.4f} | {m['R2']:.4f} |"
      )
    lines.append("")

  lines.append("Catatan, checkpoint tuned disimpan terpisah "
               f"(`{TUNED_CHECKPOINT}`), tidak menimpa checkpoint Stage 05. "
               "Kalau hasil di atas lebih baik, adopsi sebagai baseline baru "
               "adalah keputusan manual terpisah.")

  path = f"{STAT_DIR}/{filename}"
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + "\n")
  print(f"  [+] Saved: {path}")


def plot_optimization_history(study, filename):
  trial_nums = [t.number for t in study.trials if t.value is not None]
  values = [t.value for t in study.trials if t.value is not None]
  best_so_far = np.minimum.accumulate(values)

  fig, ax = plt.subplots(figsize=(9, 5))
  ax.scatter(trial_nums, values, s=18, alpha=0.5, color='#4C72B0', label='Trial val loss')
  ax.plot(trial_nums, best_so_far, color='#C44E52', linewidth=1.5, label='Best so far')
  ax.set_title('MLP Optimization History', fontsize=12, fontweight='bold')
  ax.set_xlabel('Trial', fontsize=10)
  ax.set_ylabel('Validation Loss (MSE)', fontsize=10)
  ax.legend(fontsize=9)
  ax.grid(True, linestyle='--', alpha=0.3)
  fig.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  fig.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close(fig)
  print(f"  [+] Saved: {path}")


def plot_metrics_comparison(metrics_base, metrics_tuned, split, filename):
  keys = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'DA', 'MASE', 'R2']
  labels = ['MAE (USD)', 'MAPE (%)', 'SMAPE (%)', 'RMSE (USD)', 'DA (%)', 'MASE', 'R2']
  fig, axes = plt.subplots(3, 3, figsize=(15, 12))
  axes = axes.flatten()
  for idx, (key, label) in enumerate(zip(keys, labels)):
    ax = axes[idx]
    vals = [metrics_base[split][key], metrics_tuned[split][key]]
    bars = ax.bar(['Baseline', 'Tuned'], vals, color=['#2C7FB8', '#55A868'], edgecolor='white')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    offset = max(abs(v) for v in vals) * 0.02 if any(vals) else 0.01
    for bar, val in zip(bars, vals):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
              f'{val:.3f}', ha='center', va='bottom', fontsize=8)
  for ax in axes[len(keys):]:
    ax.set_visible(False)
  fig.suptitle(f'MLP, Baseline vs Tuned, {split.capitalize()} Set',
               fontsize=14, fontweight='bold')
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


# ============================================================
# SECTION 8: PIPELINE
# ============================================================

def main():
  os.makedirs(STAT_DIR, exist_ok=True)
  log_path = f"{STAT_DIR}/00_log.txt"
  log_file = open(log_path, 'w', encoding='utf-8')
  original_stdout = sys.stdout
  sys.stdout = Tee(original_stdout, log_file)
  try:
    print(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _run_pipeline()
    print(f"\nLog finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  finally:
    sys.stdout = original_stdout
    log_file.close()
    print(f"[+] Full log saved: {log_path}")


def _run_pipeline():
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("=" * 66)
  print(f"  STAGE 11: HYPERPARAMETER TUNING (MLP), run={TAG}")
  print(f"  Device: {device} | Seed: {SEED} | Trials: {N_TRIALS}")
  print("=" * 66)

  for d in [STAT_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

  print("\n[*] Loading data...")
  splits, scaler_X, scaler_y = load_data()
  split_keys = [k for k in ["test", "unseen"] if k in splits]

  X_train_full, y_train_full = splits["train"]
  n_val = int(len(X_train_full) * VAL_FRACTION)
  n_tr = len(X_train_full) - n_val

  # MLP uses only the most-recent timestep (t-0, i.e. lag_1) of each window,
  # same convention as Stage 05.
  X_tr_flat = X_train_full[:n_tr, -1, :]
  y_tr = y_train_full[:n_tr]
  X_val_flat = X_train_full[n_tr:, -1, :]
  y_val = y_train_full[n_tr:]

  train_ld = make_loader(X_tr_flat, y_tr, BATCH_SIZE, shuffle=True)
  val_ld = make_loader(X_val_flat, y_val, BATCH_SIZE)

  def inv_y(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

  y_true_inv = {sk: inv_y(splits[sk][1]) for sk in split_keys}
  prev_price = {sk: prev_price_inv(splits[sk][0][:, -1, :], scaler_X) for sk in split_keys}
  X_flat = {sk: splits[sk][0][:, -1, :] for sk in split_keys}

  # ----------------------------------------------------------
  # Optuna search
  # ----------------------------------------------------------
  print(f"\n[*] Running Optuna search ({N_TRIALS} trials)...")
  sampler = TPESampler(seed=SEED)
  pruner = MedianPruner(n_warmup_steps=10)
  study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
  study.optimize(
      lambda trial: objective_mlp(trial, train_ld, val_ld, device),
      n_trials=N_TRIALS, show_progress_bar=False
  )
  best_params = study.best_params
  best_val_loss = study.best_value
  print(f"  [+] Best val loss: {best_val_loss:.6f}")
  print(f"  [+] Best params: {json.dumps(best_params)}")

  # ----------------------------------------------------------
  # Final retrain at full epoch budget
  # ----------------------------------------------------------
  print(f"\n[*] Retraining MLP with best params (max_epochs={MAX_EPOCHS})...")
  tuned_model = MLPModelTunable(
      best_params['h1'], best_params['h2'], best_params['h3'], best_params['dropout'])
  tuned_model, _ = train_model(
      tuned_model, train_ld, val_ld, best_params['lr'], device,
      weight_decay=best_params['weight_decay'], max_epochs=MAX_EPOCHS, patience=PATIENCE
  )
  torch.save(tuned_model.state_dict(), TUNED_CHECKPOINT)
  print(f"  [+] Saved: {TUNED_CHECKPOINT}")

  # ----------------------------------------------------------
  # Baseline (Stage 05 checkpoint, unchanged)
  # ----------------------------------------------------------
  print("\n[*] Loading baseline MLP checkpoint (Stage 05, unchanged)...")
  baseline_model = MLPModelBaseline().to(device)
  baseline_model.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=device))
  baseline_model.eval()

  # ----------------------------------------------------------
  # Metrics
  # ----------------------------------------------------------
  print("\n[*] Computing metrics...")
  metrics_base, metrics_tuned = {}, {}
  for label, model, metrics_dict in [("baseline", baseline_model, metrics_base),
                                     ("tuned", tuned_model, metrics_tuned)]:
    for sk in split_keys:
      p_inv = scaler_y.inverse_transform(
          predict_array(model, X_flat[sk], device).reshape(-1, 1)).flatten()
      metrics_dict[sk] = compute_metrics(y_true_inv[sk], p_inv, prev_price[sk])
    parts = " | ".join(
        f"{sk.upper()}: MAE={metrics_dict[sk]['MAE']:.4f} MAPE={metrics_dict[sk]['MAPE']:.4f}%"
        for sk in split_keys
    )
    print(f"  [{label}] {parts}")

  # ----------------------------------------------------------
  # Reports
  # ----------------------------------------------------------
  print("\n[*] Saving reports...")
  save_tuning_report(best_params, best_val_loss, metrics_base, metrics_tuned,
                     split_keys, "01_mlp_tuning_summary.md")
  plot_optimization_history(study, "02_optimization_history.png")
  for sk in split_keys:
    plot_metrics_comparison(metrics_base, metrics_tuned, sk,
                            f"03_metrics_comparison_{sk}.png")

  print(f"\n[+] Done. Run tag: {TAG}. Outputs in {STAT_DIR} and {GRAPH_DIR}.")


if __name__ == "__main__":
  main()
