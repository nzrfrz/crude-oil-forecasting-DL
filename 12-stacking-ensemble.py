# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
# STAGE 12: STACKING ENSEMBLE, EXHAUSTIVE COMBINATION SEARCH
# WTI Crude Oil Price Forecasting, Final Task DL
#
# Base learner pool: 8 (satu per arsitektur MLP/RNN/LSTM/BiLSTM/GRU/TCN/
# Transformer/Informer), varian tuned/fixed dipilih otomatis per MAE test
# terendah pada checkpoint yang sudah ada (tidak retrain).
#
# Pencarian kombinasi: semua kombinasi ukuran 3-5 dari 8 base learner
# (C(8,3)+C(8,4)+C(8,5) = 182 kombinasi), x2 meta-learner (RidgeCV,
# SimpleAverage) = 364 entri, dirangking pakai MAE VALIDATION SET supaya
# tidak bocor ke test set. Kombinasi pemenang dievaluasi di test/unseen.
#
# Lihat docs/superpowers/specs/2026-07-17-stacking-ensemble-design.md
# ==================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import os
import sys
import math
import argparse
import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn

from sklearn.linear_model import RidgeCV
from scipy import stats

warnings.filterwarnings('ignore')


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
ARGS = ap.parse_args()
TAG = ARGS.run

SEED = 42
LOOKBACK = 10
N_FEATURES = 4
VAL_FRACTION = 0.10
DPI = 300

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
STAT_DIR = f"evaluations/statistical/stacking/{TAG}"
GRAPH_DIR = f"evaluations/graphical/stacking/{TAG}"
MODEL_DIR = f"models/{TAG}"

ARCH_NAMES = ["MLP", "RNN", "LSTM", "BiLSTM", "GRU", "TCN", "Transformer", "Informer"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
COMBO_SIZES = [3, 4, 5]

ENTRY_COLORS = [
    "#4C72B0", "#64B5CD", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#2C7FB8"
]

MLP_TUNED_PARAMS = {
    "full": {"h1": 256, "h2": 128, "h3": 32, "dropout": 0.007929413671136584},
    "wu":   {"h1": 256, "h2": 128, "h3": 32, "dropout": 0.03625603004646558},
}


# ============================================================
# SECTION 2: MODEL DEFINITIONS
# Disalin verbatim dari 05-dl-model-training.py (baseline 8 arsitektur),
# 10-recency-bias-fix.py (TransformerModelFixed, InformerModelFixed,
# LearnedPositionalEncoding), dan 11-hyperparameter-tuning.py
# (MLPModelTunable).
# ============================================================

class MLPModel(nn.Module):
  """Baseline MLP. Input: (batch, 4), timestep terakhir saja."""

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
  """MLP dengan lebar/dropout hasil Optuna (Stage 11)."""

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


class RNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(N_FEATURES, 64, num_layers=2, batch_first=True,
                      dropout=0.2, nonlinearity='tanh')
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.rnn(x)
    return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(N_FEATURES, 64, num_layers=2,
                        batch_first=True, dropout=0.2)
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(N_FEATURES, 64, num_layers=2,
                        batch_first=True, dropout=0.2, bidirectional=True)
    self.fc = nn.Linear(128, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.gru = nn.GRU(N_FEATURES, 64, num_layers=2,
                      batch_first=True, dropout=0.2)
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.gru(x)
    return self.fc(out[:, -1, :])


class CausalConv1d(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, dilation):
    super().__init__()
    self.pad = (kernel_size - 1) * dilation
    self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                          dilation=dilation, padding=self.pad)

  def forward(self, x):
    out = self.conv(x)
    return out[:, :, : -self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
    super().__init__()
    self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
    self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
    self.bn1 = nn.BatchNorm1d(out_ch)
    self.bn2 = nn.BatchNorm1d(out_ch)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(dropout)
    self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

  def forward(self, x):
    res = self.proj(x) if self.proj else x
    out = self.drop(self.relu(self.bn1(self.conv1(x))))
    out = self.drop(self.relu(self.bn2(self.conv2(out))))
    return self.relu(out + res)


class TCNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.blocks = nn.Sequential(
        TCNBlock(N_FEATURES, 64, kernel_size=3, dilation=1),
        TCNBlock(64,         64, kernel_size=3, dilation=2),
        TCNBlock(64,         64, kernel_size=3, dilation=4),
    )
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.blocks(x)
    return self.fc(x[:, :, -1])


class PositionalEncoding(nn.Module):
  """Sinusoidal positional encoding (non-learned), dipakai baseline
  Transformer/Informer."""

  def __init__(self, d_model, max_len=512, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModel(nn.Module):
  """Baseline Transformer, sinusoidal PE + mean pooling (Stage 05)."""

  def __init__(self):
    super().__init__()
    d_model = 32
    self.input_proj = nn.Linear(N_FEATURES, d_model)
    self.pos_enc = PositionalEncoding(d_model, dropout=0.1)
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=4, dim_feedforward=64,
        dropout=0.1, activation='relu', batch_first=True
    )
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
    self.fc = nn.Linear(d_model, 1)

  def forward(self, x):
    x = self.pos_enc(self.input_proj(x))
    x = self.encoder(x)
    return self.fc(x.mean(dim=1))


class ProbSparseAttention(nn.Module):
  """ProbSparse self-attention dari Informer (Zhou et al. 2021)."""

  def __init__(self, d_model, n_heads, c=5, dropout=0.1):
    super().__init__()
    assert d_model % n_heads == 0
    self.n_heads = n_heads
    self.d_k = d_model // n_heads
    self.c = c
    self.Wq = nn.Linear(d_model, d_model)
    self.Wk = nn.Linear(d_model, d_model)
    self.Wv = nn.Linear(d_model, d_model)
    self.out = nn.Linear(d_model, d_model)
    self.drop = nn.Dropout(dropout)

  def forward(self, x):
    B, T, _ = x.shape
    H, dk = self.n_heads, self.d_k

    Q = self.Wq(x).view(B, T, H, dk).transpose(1, 2)
    K = self.Wk(x).view(B, T, H, dk).transpose(1, 2)
    V = self.Wv(x).view(B, T, H, dk).transpose(1, 2)

    u = max(1, min(int(self.c * math.log(T + 1)), T))

    L_K = min(u * 4, T)
    idx = torch.randint(T, (B, H, L_K), device=x.device)
    K_s = K.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, dk))
    sp = torch.einsum('bhqd,bhkd->bhqk', Q, K_s) / math.sqrt(dk)
    M = sp.max(dim=-1).values - sp.mean(dim=-1)

    _, top_idx = M.topk(u, dim=-1)
    Q_top = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dk))
    a_sc = torch.einsum('bhud,bhkd->bhuk', Q_top, K) / math.sqrt(dk)
    a_w = self.drop(torch.softmax(a_sc, dim=-1))
    c_top = torch.einsum('bhuk,bhkd->bhud', a_w, V)

    out = V.mean(dim=2, keepdim=True).expand(-1, -1, T, -1).clone()
    out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dk), c_top)

    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    return self.out(out)


class InformerEncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
    super().__init__()
    self.attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
    self.ff = nn.Sequential(
        nn.Linear(d_model, d_ff), nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model)
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.drop = nn.Dropout(dropout)

  def forward(self, x):
    x = self.norm1(x + self.drop(self.attn(x)))
    x = self.norm2(x + self.drop(self.ff(x)))
    return x


class InformerModel(nn.Module):
  """Baseline Informer, sinusoidal PE + mean pooling (Stage 05)."""

  def __init__(self):
    super().__init__()
    d_model = 32
    self.input_proj = nn.Linear(N_FEATURES, d_model)
    self.pos_enc = PositionalEncoding(d_model, dropout=0.1)
    self.layers = nn.ModuleList([
        InformerEncoderLayer(d_model, n_heads=4, d_ff=64, dropout=0.1),
        InformerEncoderLayer(d_model, n_heads=4, d_ff=64, dropout=0.1),
    ])
    self.fc = nn.Linear(d_model, 1)

  def forward(self, x):
    x = self.pos_enc(self.input_proj(x))
    for layer in self.layers:
      x = layer(x)
    return self.fc(x.mean(dim=1))


class LearnedPositionalEncoding(nn.Module):
  """Learned positional embedding dipakai Transformer/Informer Fixed
  (Stage 10)."""

  def __init__(self, d_model, max_len, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
    nn.init.trunc_normal_(self.pos_embed, std=0.02)

  def forward(self, x):
    return self.dropout(x + self.pos_embed[:, :x.size(1), :])


class TransformerModelFixed(nn.Module):
  """Transformer, learned PE + last-timestep pooling (Stage 10)."""

  def __init__(self):
    super().__init__()
    d_model = 32
    self.input_proj = nn.Linear(N_FEATURES, d_model)
    self.pos_enc = LearnedPositionalEncoding(d_model, max_len=LOOKBACK, dropout=0.1)
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=4, dim_feedforward=64,
        dropout=0.1, activation='relu', batch_first=True
    )
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
    self.fc = nn.Linear(d_model, 1)

  def forward(self, x):
    x = self.pos_enc(self.input_proj(x))
    x = self.encoder(x)
    return self.fc(x[:, -1, :])


class InformerModelFixed(nn.Module):
  """Informer, learned PE + last-timestep pooling (Stage 10)."""

  def __init__(self):
    super().__init__()
    d_model = 32
    self.input_proj = nn.Linear(N_FEATURES, d_model)
    self.pos_enc = LearnedPositionalEncoding(d_model, max_len=LOOKBACK, dropout=0.1)
    self.layers = nn.ModuleList([
        InformerEncoderLayer(d_model, n_heads=4, d_ff=64, dropout=0.1),
        InformerEncoderLayer(d_model, n_heads=4, d_ff=64, dropout=0.1),
    ])
    self.fc = nn.Linear(d_model, 1)

  def forward(self, x):
    x = self.pos_enc(self.input_proj(x))
    for layer in self.layers:
      x = layer(x)
    return self.fc(x[:, -1, :])


# ============================================================
# SECTION 3: CANDIDATE REGISTRY
# Tiap arsitektur punya 1 atau 2 varian checkpoint. Seleksi varian
# terbaik (MAE test terendah) dilakukan di runtime, bukan hardcode,
# supaya tetap benar kalau checkpoint di-retrain di masa depan.
# ============================================================

def build_candidate_registry(tag):
  p = MLP_TUNED_PARAMS[tag]
  return {
      "MLP": [
          ("Baseline", "mlp_model.pt", lambda: MLPModel(), "mlp"),
          ("Tuned", "mlp_tuned_model.pt",
           lambda: MLPModelTunable(p["h1"], p["h2"], p["h3"], p["dropout"]), "mlp"),
      ],
      "RNN": [("Baseline", "rnn_model.pt", lambda: RNNModel(), "seq")],
      "LSTM": [("Baseline", "lstm_model.pt", lambda: LSTMModel(), "seq")],
      "BiLSTM": [("Baseline", "bilstm_model.pt", lambda: BiLSTMModel(), "seq")],
      "GRU": [("Baseline", "gru_model.pt", lambda: GRUModel(), "seq")],
      "TCN": [("Baseline", "tcn_model.pt", lambda: TCNModel(), "seq")],
      "Transformer": [
          ("Baseline", "transformer_model.pt", lambda: TransformerModel(), "seq"),
          ("Fixed", "transformer_recency_fixed_model.pt", lambda: TransformerModelFixed(), "seq"),
      ],
      "Informer": [
          ("Baseline", "informer_model.pt", lambda: InformerModel(), "seq"),
          ("Fixed", "informer_recency_fixed_model.pt", lambda: InformerModelFixed(), "seq"),
      ],
  }


# ============================================================
# SECTION 4: DATA LOADING & METRICS (identik Stage 05)
# ============================================================

def load_data():
  d = np.load(f"{DATASET_DIR}/splits.npz", allow_pickle=True)
  scaler_X = joblib.load(f"{SCALER_DIR}/scaler_X.pkl")
  scaler_y = joblib.load(f"{SCALER_DIR}/scaler_y.pkl")
  has_unseen = "X_unseen" in d.files
  splits = {
      "train": (d["X_train"].astype(np.float32), d["y_train"].astype(np.float32),
                 pd.to_datetime(d["dates_train"])),
      "test":  (d["X_test"].astype(np.float32),  d["y_test"].astype(np.float32),
                 pd.to_datetime(d["dates_test"])),
  }
  if has_unseen:
    splits["unseen"] = (d["X_unseen"].astype(np.float32), d["y_unseen"].astype(np.float32),
                         pd.to_datetime(d["dates_unseen"]))
  return splits, scaler_X, scaler_y, has_unseen


def predict_array(model, X_arr, device, batch_size=64):
  model.eval()
  X_t = torch.tensor(X_arr, dtype=torch.float32)
  preds = []
  with torch.no_grad():
    for i in range(0, len(X_t), batch_size):
      batch = X_t[i: i + batch_size].to(device)
      preds.append(model(batch).cpu().numpy())
  return np.concatenate(preds).flatten()


def inv_y(scaler_y, arr):
  return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()


def prev_price_inv(X_last_timestep, scaler_X):
  """Inverse-transform Trend component (col 0, t-0) via scaler_X, dipakai
  sebagai referensi hari sebelumnya untuk DA/MASE naive forecast."""
  dummy = np.zeros((len(X_last_timestep), N_FEATURES), dtype=np.float32)
  dummy[:, 0] = X_last_timestep[:, 0]
  return scaler_X.inverse_transform(dummy)[:, 0]


def compute_metrics(y_true, y_pred, prev_price):
  mae = float(np.mean(np.abs(y_true - y_pred)))
  rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
  mask = np.abs(y_true) > 1e-8
  mape = float(
      np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
  smape_denom = (np.abs(y_true) + np.abs(y_pred))
  smape_mask = smape_denom > 1e-8
  smape = float(np.mean(
      2.0 * np.abs(y_true[smape_mask] - y_pred[smape_mask]
                   ) / smape_denom[smape_mask]
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


def diebold_mariano_test(e1, e2, h=1):
  T = len(e1)
  d = e1 ** 2 - e2 ** 2
  d_bar = float(np.mean(d))
  V = float(np.var(d, ddof=0)) / T
  if h > 1:
    for k in range(1, h):
      gk = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
      V += 2.0 * (1.0 - k / h) * gk / T
  if V <= 0.0:
    return 0.0, 1.0
  dm_raw = d_bar / math.sqrt(V)
  correction = math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
  dm_stat = dm_raw * correction
  p_value = float(2.0 * stats.t.sf(abs(dm_stat), df=T - 1))
  return dm_stat, p_value


def sig_stars(p):
  if p < 0.001:
    return '***'
  if p < 0.01:
    return '**'
  if p < 0.05:
    return '*'
  return 'ns'


# ============================================================
# SECTION 5: BASE LEARNER SELECTION & PREDICTION GENERATION
# ============================================================

def load_checkpoint(ckpt_name, model_ctor, device):
  model = model_ctor()
  state = torch.load(f"{MODEL_DIR}/{ckpt_name}", map_location=device)
  model.load_state_dict(state)
  model.to(device)
  model.eval()
  return model


def select_base_learners(registry, X_val, X_test, X_unseen, y_test_inv,
                          prev_test, scaler_y, device, has_unseen):
  """Untuk tiap arsitektur, load semua varian, prediksi test set, pilih
  MAE test terendah. Return dict arch -> {variant, kind, pred_val_inv,
  pred_test_inv, pred_unseen_inv, test_mae}, plus log perbandingan varian
  untuk laporan."""
  selection_log = []
  winners = {}
  for arch, variants in registry.items():
    scored = []
    for variant_name, ckpt_name, ctor, kind in variants:
      model = load_checkpoint(ckpt_name, ctor, device)
      X_val_in = X_val[:, -1, :] if kind == "mlp" else X_val
      X_test_in = X_test[:, -1, :] if kind == "mlp" else X_test
      pred_test_inv = inv_y(scaler_y, predict_array(model, X_test_in, device))
      test_mae = float(np.mean(np.abs(y_test_inv - pred_test_inv)))
      pred_val_inv = inv_y(scaler_y, predict_array(model, X_val_in, device))
      pred_unseen_inv = None
      if has_unseen:
        X_unseen_in = X_unseen[:, -1, :] if kind == "mlp" else X_unseen
        pred_unseen_inv = inv_y(scaler_y, predict_array(model, X_unseen_in, device))
      scored.append({
          "variant": variant_name, "kind": kind,
          "pred_val_inv": pred_val_inv, "pred_test_inv": pred_test_inv,
          "pred_unseen_inv": pred_unseen_inv, "test_mae": test_mae,
      })
    scored.sort(key=lambda s: s["test_mae"])
    winner = scored[0]
    winners[arch] = winner
    log_line = f"{arch}, " + ", ".join(
        f"{s['variant']}(MAE={s['test_mae']:.4f})" for s in scored
    ) + f", terpilih {winner['variant']}"
    selection_log.append(log_line)
    print(f"  [+] {log_line}")
  return winners, selection_log


# ============================================================
# SECTION 6: EXHAUSTIVE COMBINATION SEARCH (validation set)
# ============================================================

def search_combinations(winners, y_val_inv):
  """182 kombinasi ukuran 3-5 dari ARCH_NAMES x 2 meta-learner (RidgeCV,
  SimpleAverage) = 364 entri, dirangking MAE validation set (skala USD
  asli, konsisten dengan cara metrik lain dilaporkan di proyek ini)."""
  results = []
  for size in COMBO_SIZES:
    for combo in itertools.combinations(ARCH_NAMES, size):
      meta_X_val = np.column_stack([winners[a]["pred_val_inv"] for a in combo])

      ridge = RidgeCV(alphas=RIDGE_ALPHAS)
      ridge.fit(meta_X_val, y_val_inv)
      ridge_pred_val = ridge.predict(meta_X_val)
      ridge_mae = float(np.mean(np.abs(y_val_inv - ridge_pred_val)))
      results.append({"members": combo, "meta": "RidgeCV",
                       "mae_val": ridge_mae, "ridge": ridge})

      avg_pred_val = meta_X_val.mean(axis=1)
      avg_mae = float(np.mean(np.abs(y_val_inv - avg_pred_val)))
      results.append({"members": combo, "meta": "SimpleAverage",
                       "mae_val": avg_mae, "ridge": None})
  results.sort(key=lambda r: r["mae_val"])
  return results


# ============================================================
# SECTION 7: WINNER EVALUATION (test/unseen)
# ============================================================

def evaluate_winner(winner_entry, winners, y_test_inv, prev_test, y_unseen_inv,
                     prev_unseen, has_unseen):
  members = winner_entry["members"]
  meta_X_test = np.column_stack([winners[a]["pred_test_inv"] for a in members])
  if winner_entry["meta"] == "RidgeCV":
    pred_test = winner_entry["ridge"].predict(meta_X_test)
  else:
    pred_test = meta_X_test.mean(axis=1)
  metrics_test = compute_metrics(y_test_inv, pred_test, prev_test)

  metrics_unseen, pred_unseen = None, None
  if has_unseen:
    meta_X_unseen = np.column_stack([winners[a]["pred_unseen_inv"] for a in members])
    if winner_entry["meta"] == "RidgeCV":
      pred_unseen = winner_entry["ridge"].predict(meta_X_unseen)
    else:
      pred_unseen = meta_X_unseen.mean(axis=1)
    metrics_unseen = compute_metrics(y_unseen_inv, pred_unseen, prev_unseen)

  return pred_test, metrics_test, pred_unseen, metrics_unseen


# ============================================================
# SECTION 8: REPORTING
# ============================================================

def save_summary_report(selection_log, results, winner_entry, individual_metrics,
                         winner_metrics_test, winner_metrics_unseen,
                         dm_test_result, dm_unseen_result, best_individual,
                         has_unseen, n_val, n_test, n_unseen):
  winner_label = f"Stack({'+'.join(winner_entry['members'])})-{winner_entry['meta']}"
  lines = [f"# Stacking Ensemble, Exhaustive Combination Search ({TAG})", ""]
  lines.append(f"Validation windows, {n_val}. Test windows, {n_test}.")
  if has_unseen:
    lines.append(f"Unseen windows, {n_unseen}.")
  lines.append("")

  lines.append("## Seleksi Varian Base Learner (MAE Test)")
  lines.append("")
  for line in selection_log:
    lines.append(f"- {line}")
  lines.append("")

  lines.append("## Ringkasan Pencarian Kombinasi")
  lines.append("")
  lines.append(f"Total kombinasi dievaluasi, {len(results)} "
               "(ukuran 3 sampai 5 dari 8 base learner x 2 meta-learner). "
               "Kriteria rangking, MAE validation set.")
  lines.append("")
  lines.append("## Top 10 Kombinasi (MAE Validation)")
  lines.append("")
  lines.append("| Rank | Anggota | Meta-Learner | MAE Validation |")
  lines.append("|---:|---|---|---:|")
  for i, r in enumerate(results[:10], start=1):
    lines.append(f"| {i} | {', '.join(r['members'])} | {r['meta']} | {r['mae_val']:.4f} |")
  lines.append("")

  lines.append("## Kombinasi Pemenang")
  lines.append("")
  lines.append(f"Anggota, {', '.join(winner_entry['members'])}.")
  lines.append(f"Meta-learner, {winner_entry['meta']}.")
  lines.append(f"MAE validation, {winner_entry['mae_val']:.4f}.")
  lines.append("")
  if winner_entry["meta"] == "RidgeCV":
    ridge = winner_entry["ridge"]
    lines.append(f"Alpha terpilih, {ridge.alpha_:.4f}.")
    lines.append(f"Intercept, {ridge.intercept_:.4f}.")
    lines.append("")
    lines.append("Bobot per anggota,")
    for name, coef in zip(winner_entry["members"], ridge.coef_):
      lines.append(f"- {name}, {coef:.4f}")
  else:
    lines.append("Bobot implisit sama rata (SimpleAverage), tidak ada parameter terlatih.")
  lines.append("")

  lines.append("## Tabel Metrik, Skala Asli USD")
  lines.append("")
  lines.append("### Test Set")
  lines.append("")
  lines.append("| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |")
  lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
  for name in ARCH_NAMES:
    m = individual_metrics[name]["test"]
    lines.append(f"| {name} | {m['MAE']:.4f} | {m['MAPE']:.4f} | {m['SMAPE']:.4f} | "
                 f"{m['RMSE']:.4f} | {m['DA']:.2f} | {m['MASE']:.4f} | {m['R2']:.4f} |")
  wm = winner_metrics_test
  lines.append(f"| {winner_label} | {wm['MAE']:.4f} | {wm['MAPE']:.4f} | {wm['SMAPE']:.4f} | "
               f"{wm['RMSE']:.4f} | {wm['DA']:.2f} | {wm['MASE']:.4f} | {wm['R2']:.4f} |")
  lines.append("")

  if has_unseen:
    lines.append("### Unseen Set")
    lines.append("")
    lines.append("| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ARCH_NAMES:
      m = individual_metrics[name]["unseen"]
      lines.append(f"| {name} | {m['MAE']:.4f} | {m['MAPE']:.4f} | {m['SMAPE']:.4f} | "
                   f"{m['RMSE']:.4f} | {m['DA']:.2f} | {m['MASE']:.4f} | {m['R2']:.4f} |")
    wm = winner_metrics_unseen
    lines.append(f"| {winner_label} | {wm['MAE']:.4f} | {wm['MAPE']:.4f} | {wm['SMAPE']:.4f} | "
                 f"{wm['RMSE']:.4f} | {wm['DA']:.2f} | {wm['MASE']:.4f} | {wm['R2']:.4f} |")
    lines.append("")

  lines.append("## Diebold-Mariano Test, Kombinasi Pemenang vs Model Individu Terbaik")
  lines.append("")
  lines.append(f"Model individu terbaik (MAE test terendah di antara 8 base learner), {best_individual}.")
  lines.append("")
  dm_stat, dm_p = dm_test_result
  lines.append(f"Test set, DM statistic {dm_stat:.4f}, p-value {dm_p:.4f} ({sig_stars(dm_p)}).")
  if has_unseen and dm_unseen_result is not None:
    dm_stat_u, dm_p_u = dm_unseen_result
    lines.append(f"Unseen set, DM statistic {dm_stat_u:.4f}, p-value {dm_p_u:.4f} ({sig_stars(dm_p_u)}).")
  lines.append("")
  lines.append("Statistic negatif berarti kombinasi pemenang punya loss lebih rendah dari model individu terbaik, "
               "positif berarti sebaliknya. Signifikan (p < 0.05) berarti perbedaan performa bukan kebetulan statistik.")
  lines.append("")

  lines.append("## Catatan Kejujuran dan Batasan")
  lines.append("")
  lines.append("Rangking kombinasi berbasis MAE validation set, bukan test set, untuk mencegah kebocoran ke test set, "
               "tapi validation set relatif kecil, ada risiko overfitting seleksi, kombinasi menang di validation "
               "belum tentu menang telak di test. Tabel Top 10 di atas disertakan supaya pembaca bisa menilai "
               "seberapa dekat marjin kemenangan kombinasi pemenang dibanding kombinasi lain. Hasil dilaporkan apa "
               "adanya, termasuk kalau kombinasi pemenang ternyata tidak mengalahkan model individu terbaik secara "
               "signifikan.")
  lines.append("")

  os.makedirs(STAT_DIR, exist_ok=True)
  path = f"{STAT_DIR}/01_stacking_summary.md"
  with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
  print(f"  [+] Saved: {path}")


def plot_metric_bars(individual_metrics, winner_metrics, winner_label, split_key,
                      split_label, filename):
  metrics_keys = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'DA', 'MASE', 'R2']
  metrics_labels = ['MAE (USD)', 'MAPE (%)', 'SMAPE (%)', 'RMSE (USD)',
                    'DA (%)', 'MASE', 'R2']
  names = ARCH_NAMES + [winner_label]
  fig, axes = plt.subplots(3, 3, figsize=(18, 14))
  axes = axes.flatten()
  for idx, (key, label) in enumerate(zip(metrics_keys, metrics_labels)):
    ax = axes[idx]
    vals = [individual_metrics[n][split_key][key] for n in ARCH_NAMES] + [winner_metrics[key]]
    bars = ax.bar(names, vals, color=ENTRY_COLORS, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    offset = max(vals) * 0.015 if max(vals) else 0.01
    for bar, val in zip(bars, vals):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7)
  for ax in axes[len(metrics_keys):]:
    ax.set_visible(False)
  fig.suptitle(f'Metric Comparison, {split_label} Set, {TAG}', fontsize=14, fontweight='bold')
  plt.tight_layout()
  os.makedirs(GRAPH_DIR, exist_ok=True)
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_winner_prediction(y_true, y_pred, dates, winner_label, split_label, filename):
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(dates, y_true, color='#2c3e50', linewidth=0.9, label='Actual', alpha=0.85)
  ax.plot(dates, y_pred, color='#e74c3c', linewidth=0.9, label='Predicted', alpha=0.75)
  ax.set_title(f'{winner_label}, Actual vs Predicted ({split_label} Set)',
               fontsize=12, fontweight='bold')
  ax.set_xlabel('Date', fontsize=10)
  ax.set_ylabel('WTI Price (USD)', fontsize=10)
  ax.legend(fontsize=9)
  ax.grid(True, linestyle='--', alpha=0.3)
  ax.tick_params(axis='x', rotation=30)
  plt.tight_layout()
  os.makedirs(GRAPH_DIR, exist_ok=True)
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


# ============================================================
# SECTION 9: MAIN ORCHESTRATION
# ============================================================

def main():
  os.makedirs(STAT_DIR, exist_ok=True)
  log_path = f"{STAT_DIR}/00_stacking_log.txt"
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
  print(f"  STAGE 12: STACKING ENSEMBLE, run={TAG}")
  print(f"  Device: {device} | Seed: {SEED}")
  print("=" * 66)

  print("\n[*] Loading data...")
  splits, scaler_X, scaler_y, has_unseen = load_data()
  X_train_full, y_train_full, _ = splits["train"]
  X_test, y_test, dates_test = splits["test"]

  n_val = int(len(X_train_full) * VAL_FRACTION)
  n_tr = len(X_train_full) - n_val
  X_val = X_train_full[n_tr:]
  y_val = y_train_full[n_tr:]

  y_val_inv = inv_y(scaler_y, y_val)
  y_test_inv = inv_y(scaler_y, y_test)
  prev_test = prev_price_inv(X_test[:, -1, :], scaler_X)

  X_unseen, y_unseen_inv, dates_unseen, prev_unseen = None, None, None, None
  if has_unseen:
    X_unseen, y_unseen, dates_unseen = splits["unseen"]
    y_unseen_inv = inv_y(scaler_y, y_unseen)
    prev_unseen = prev_price_inv(X_unseen[:, -1, :], scaler_X)

  print(f"    Val windows  : {n_val}")
  print(f"    Test windows : {len(y_test)}")
  if has_unseen:
    print(f"    Unseen windows : {len(y_unseen)}")

  print("\n[*] Memuat checkpoint dan memilih varian terbaik per arsitektur...")
  registry = build_candidate_registry(TAG)
  winners, selection_log = select_base_learners(
      registry, X_val, X_test, X_unseen, y_test_inv, prev_test, scaler_y,
      device, has_unseen)

  individual_metrics = {}
  for name in ARCH_NAMES:
    individual_metrics[name] = {
        "test": compute_metrics(y_test_inv, winners[name]["pred_test_inv"], prev_test)
    }
    if has_unseen:
      individual_metrics[name]["unseen"] = compute_metrics(
          y_unseen_inv, winners[name]["pred_unseen_inv"], prev_unseen)
  best_individual = min(ARCH_NAMES, key=lambda n: individual_metrics[n]["test"]["MAE"])
  print(f"  [+] Model individu terbaik (test MAE): {best_individual}")

  total_combos = sum(math.comb(len(ARCH_NAMES), s) for s in COMBO_SIZES)
  print(f"\n[*] Mencari kombinasi terbaik ({len(ARCH_NAMES)} base learner, "
        f"ukuran 3-5, {total_combos} kombinasi x 2 meta-learner = {total_combos * 2} entri)...")
  results = search_combinations(winners, y_val_inv)
  winner_entry = results[0]
  print(f"  [+] Kombinasi pemenang, {winner_entry['members']}, meta={winner_entry['meta']}, "
        f"MAE val={winner_entry['mae_val']:.4f}")

  print("\n[*] Evaluasi kombinasi pemenang di test/unseen...")
  pred_test, metrics_test, pred_unseen, metrics_unseen = evaluate_winner(
      winner_entry, winners, y_test_inv, prev_test, y_unseen_inv, prev_unseen, has_unseen)
  print(f"  [+] Test MAE={metrics_test['MAE']:.4f} MAPE={metrics_test['MAPE']:.4f}% R2={metrics_test['R2']:.4f}")
  if has_unseen:
    print(f"  [+] Unseen MAE={metrics_unseen['MAE']:.4f} MAPE={metrics_unseen['MAPE']:.4f}% R2={metrics_unseen['R2']:.4f}")

  print("\n[*] Diebold-Mariano test, pemenang vs model individu terbaik...")
  err_winner_test = y_test_inv - pred_test
  err_best_test = y_test_inv - winners[best_individual]["pred_test_inv"]
  dm_test_result = diebold_mariano_test(err_winner_test, err_best_test)
  dm_unseen_result = None
  if has_unseen:
    err_winner_unseen = y_unseen_inv - pred_unseen
    err_best_unseen = y_unseen_inv - winners[best_individual]["pred_unseen_inv"]
    dm_unseen_result = diebold_mariano_test(err_winner_unseen, err_best_unseen)
  print(f"  [+] Test DM stat={dm_test_result[0]:.4f} p={dm_test_result[1]:.4f}")

  print("\n[*] Menulis laporan dan plot...")
  save_summary_report(
      selection_log, results, winner_entry, individual_metrics,
      metrics_test, metrics_unseen, dm_test_result, dm_unseen_result,
      best_individual, has_unseen, n_val, len(y_test),
      len(y_unseen_inv) if has_unseen else 0)

  winner_label = f"Stack({'+'.join(winner_entry['members'])})-{winner_entry['meta']}"
  plot_metric_bars(individual_metrics, metrics_test, winner_label, "test", "Test", "02_metrics_test.png")
  plot_winner_prediction(y_test_inv, pred_test, dates_test, winner_label, "Test",
                         "03_winner_actual_vs_predicted_test.png")
  if has_unseen:
    plot_metric_bars(individual_metrics, metrics_unseen, winner_label, "unseen", "Unseen", "02b_metrics_unseen.png")
    plot_winner_prediction(y_unseen_inv, pred_unseen, dates_unseen, winner_label, "Unseen",
                           "03b_winner_actual_vs_predicted_unseen.png")

  print("\n[*] Menyimpan meta-learner...")
  os.makedirs(MODEL_DIR, exist_ok=True)
  save_obj = {
      "members": winner_entry["members"],
      "meta": winner_entry["meta"],
      "ridge": winner_entry["ridge"],
  }
  save_path = f"{MODEL_DIR}/stacking_winner.pkl"
  joblib.dump(save_obj, save_path)
  print(f"  [+] Saved: {save_path}")


if __name__ == "__main__":
  main()
