# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
# STAGE 05: DEEP LEARNING MODEL TRAINING & COMPARISON
# WTI Crude Oil Price Forecasting, CEEMDAN 4-component input (Final Task DL)
# Models: MLP, RNN, LSTM, BiLSTM, GRU, TCN, Transformer, Informer
# Metrics: MAE, MAPE, SMAPE, RMSE, Directional Accuracy, MASE, R2
# Statistical Test: Diebold-Mariano (Harvey et al. 1997)
#
# Adaptasi dari D:\Coding\#bigdata\crude-oil-forecasting-DL\05-dl-model-training.py
# dan 08-rnn-model-training.py (kelas RNNModel). Kelas model dan fungsi inti
# disalin verbatim, hanya input diganti dari 7 fitur makro menjadi 4 komponen
# CEEMDAN (Trend, group_1, group_2, res), dan data sudah pre-windowed
# (dataset/splits/<tag>/splits.npz dari 04-fe-and-split.py), jadi tidak perlu
# create_sequences/create_eval_sequences seperti sumber.
# ==================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import os
import sys
import argparse
import warnings
import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

warnings.filterwarnings('ignore')


class Tee:
  """Mirrors every write to multiple streams (e.g. console + log file),
  flushing immediately so a crash/power-loss only drops unwritten output."""

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
ap.add_argument("--max-epochs", type=int, default=200)
ARGS = ap.parse_args()
TAG = ARGS.run

SEED = 42
LOOKBACK = 10          # T: sliding window length (trading days)
N_FEATURES = 4           # number of input CEEMDAN components
BATCH_SIZE = 32
MAX_EPOCHS = ARGS.max_epochs
PATIENCE = 20          # early stopping patience
LR = 1e-3        # Adam initial learning rate
LR_FACTOR = 0.5         # ReduceLROnPlateau reduction factor
LR_PATIENCE = 10          # epochs without improvement before LR reduction
LR_MIN = 1e-5        # minimum learning rate floor
VAL_FRACTION = 0.10       # fraction of training windows used as validation
DPI = 300
N_PERM_REPEATS = 10       # permutation repeats per feature for stable importance estimate

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
STAT_DIR = f"evaluations/statistical/model-train/{TAG}"
GRAPH_DIR = f"evaluations/graphical/model-train/{TAG}"
MODEL_DIR = f"models/{TAG}"

PRED_TEST_DIR = f"{GRAPH_DIR}/actual-vs-predicted-test-set"
PRED_UNSEEN_DIR = f"{GRAPH_DIR}/actual-vs-predicted-unseen-set"
FI_TEST_DIR = f"{GRAPH_DIR}/feature-importance-test-set"
FI_UNSEEN_DIR = f"{GRAPH_DIR}/feature-importance-unseen-set"

MODEL_NAMES = ["MLP", "RNN", "LSTM", "BiLSTM",
               "GRU", "TCN", "Transformer", "Informer"]

FEATURE_NAMES = ["Trend", "IMF_Group1", "IMF_Group2", "Residual"]

# Consistent color palette for model bars
MODEL_COLORS = [
    "#4C72B0", "#64B5CD", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3"
]


# ============================================================
# SECTION 2: DATA LOADING (pre-windowed, no sequence building needed)
# ============================================================

def load_data():
  """Load pre-windowed splits from 04-fe-and-split.py. Returns a dict keyed
  by split name ('train'/'test'/'unseen') -> (X (N,T,F), y (N,), dates),
  plus scaler_X, scaler_y, and has_unseen flag ('wu' run has no unseen)."""
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


def make_loader(X, y, batch_size, shuffle=False):
  X_t = torch.tensor(X, dtype=torch.float32)
  y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
  return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


# ============================================================
# SECTION 3: MODEL DEFINITIONS
# Verbatim from crude-oil-forecasting-DL/05-dl-model-training.py (lines
# 166-415) and 08-rnn-model-training.py (RNNModel, lines 164-176), only
# N_FEATURES changed (4 CEEMDAN components instead of 7 macro features).
# ============================================================

class MLPModel(nn.Module):
  """Multi-Layer Perceptron baseline. Input: (batch, 4)."""

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


class RNNModel(nn.Module):
  """Vanilla RNN (tanh nonlinearity). Same hidden/layer/dropout scale as
  LSTM/GRU for a fair architecture-family comparison."""

  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(N_FEATURES, 64, num_layers=2, batch_first=True,
                      dropout=0.2, nonlinearity='tanh')
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.rnn(x)
    return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
  """Stacked LSTM. Input: (batch, seq, 4)."""

  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(N_FEATURES, 64, num_layers=2,
                        batch_first=True, dropout=0.2)
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
  """Bidirectional LSTM. Output dim is 128 (64 fwd + 64 bwd)."""

  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(N_FEATURES, 64, num_layers=2,
                        batch_first=True, dropout=0.2, bidirectional=True)
    self.fc = nn.Linear(128, 1)

  def forward(self, x):
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
  """Stacked GRU. Input: (batch, seq, 4)."""

  def __init__(self):
    super().__init__()
    self.gru = nn.GRU(N_FEATURES, 64, num_layers=2,
                      batch_first=True, dropout=0.2)
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    out, _ = self.gru(x)
    return self.fc(out[:, -1, :])


class CausalConv1d(nn.Module):
  """
  Causal (left-only) 1D convolution via symmetric padding + right trim.
  Maintains sequence length.
  """

  def __init__(self, in_ch, out_ch, kernel_size, dilation):
    super().__init__()
    self.pad = (kernel_size - 1) * dilation
    self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                          dilation=dilation, padding=self.pad)

  def forward(self, x):
    out = self.conv(x)
    return out[:, :, : -self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
  """Residual TCN block: two causal convs + BN + ReLU + skip connection."""

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
  """
  Temporal Convolutional Network with dilated causal convolutions.
  Input: (batch, seq, 4) -> permuted to (batch, 4, seq) for Conv1d.
  Receptive field after 2 blocks: 1+2*(3-1)*1+2*(3-1)*2 = 13 > T=10.
  """

  def __init__(self):
    super().__init__()
    self.blocks = nn.Sequential(
        TCNBlock(N_FEATURES, 64, kernel_size=3, dilation=1),
        TCNBlock(64,         64, kernel_size=3, dilation=2),
        TCNBlock(64,         64, kernel_size=3, dilation=4),
    )
    self.fc = nn.Linear(64, 1)

  def forward(self, x):
    x = x.permute(0, 2, 1)       # (batch, features, seq)
    x = self.blocks(x)           # (batch, 64, seq)
    return self.fc(x[:, :, -1])  # take last timestep


class PositionalEncoding(nn.Module):
  """Sinusoidal positional encoding (non-learned)."""

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
    self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

  def forward(self, x):
    return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModel(nn.Module):
  """
  Encoder-only Transformer with mean pooling over sequence dim.
  d_model=32, nhead=4, 2 encoder layers.
  """

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
    x = self.pos_enc(self.input_proj(x))  # (batch, seq, 32)
    x = self.encoder(x)                   # (batch, seq, 32)
    return self.fc(x.mean(dim=1))         # mean pool -> (batch, 1)


class ProbSparseAttention(nn.Module):
  """
  ProbSparse self-attention from Informer (Zhou et al. 2021).
  Selects top-u active queries by approximated KL-divergence score,
  computes full attention for those, fills rest with mean(V).
  For T=10 and c=5: u = min(c*ln(T+1), T) ~= 10 -> degrades to full attention.
  """

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

    Q = self.Wq(x).view(B, T, H, dk).transpose(1, 2)  # (B,H,T,dk)
    K = self.Wk(x).view(B, T, H, dk).transpose(1, 2)
    V = self.Wv(x).view(B, T, H, dk).transpose(1, 2)

    u = max(1, min(int(self.c * math.log(T + 1)), T))

    L_K = min(u * 4, T)
    idx = torch.randint(T, (B, H, L_K), device=x.device)
    K_s = K.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, dk))
    sp = torch.einsum('bhqd,bhkd->bhqk', Q, K_s) / math.sqrt(dk)
    M = sp.max(dim=-1).values - sp.mean(dim=-1)  # (B,H,T)

    _, top_idx = M.topk(u, dim=-1)  # (B,H,u)
    Q_top = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dk))
    a_sc = torch.einsum('bhud,bhkd->bhuk', Q_top, K) / math.sqrt(dk)
    a_w = self.drop(torch.softmax(a_sc, dim=-1))
    c_top = torch.einsum('bhuk,bhkd->bhud', a_w, V)  # (B,H,u,dk)

    out = V.mean(dim=2, keepdim=True).expand(-1, -1, T, -1).clone()
    out.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, dk), c_top)

    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    return self.out(out)


class InformerEncoderLayer(nn.Module):
  """Informer encoder layer: ProbSparse attention + FFN + LayerNorm."""

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
  """
  Informer encoder-only model with ProbSparse attention.
  Same d_model/nhead as TransformerModel for fair comparison.
  """

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


# ============================================================
# SECTION 4: TRAINING INFRASTRUCTURE
# Verbatim from crude-oil-forecasting-DL/05-dl-model-training.py
# ============================================================

class EarlyStopping:
  """Monitors val loss and saves the best model weights in CPU memory."""

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
      self.best_state = {k: v.cpu().clone()
                         for k, v in model.state_dict().items()}
    else:
      self.counter += 1
    return self.counter >= self.patience

  def restore_best(self, model):
    if self.best_state:
      model.load_state_dict(self.best_state)


def train_model(model, train_loader, val_loader, name, device):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=LR_FACTOR,
      patience=LR_PATIENCE, min_lr=LR_MIN
  )
  criterion = nn.MSELoss()
  stopper = EarlyStopping(patience=PATIENCE)
  history = {'train_loss': [], 'val_loss': []}
  t0 = time.time()

  for epoch in range(1, MAX_EPOCHS + 1):
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

    if epoch == 1 or epoch % 20 == 0:
      lr_now = optimizer.param_groups[0]['lr']
      print(f"  [{name}] Epoch {epoch:3d} | "
            f"Train={tr_loss:.6f} | Val={va_loss:.6f} | LR={lr_now:.2e}")

    if stopper.step(va_loss, model):
      print(f"  [{name}] Early stop @ epoch {epoch} "
            f"(best_val={stopper.best_loss:.6f})")
      break

  stopper.restore_best(model)
  elapsed = time.time() - t0
  print(f"  [{name}] Finished in {elapsed:.1f}s | "
        f"Best val: {stopper.best_loss:.6f}")
  return model, history, elapsed


# ============================================================
# SECTION 5: EVALUATION UTILITIES
# Verbatim from crude-oil-forecasting-DL/05-dl-model-training.py
# ============================================================

def predict(model, loader, device):
  model.eval()
  preds, trues = [], []
  with torch.no_grad():
    for X_b, y_b in loader:
      preds.append(model(X_b.to(device)).cpu().numpy())
      trues.append(y_b.numpy())
  return np.concatenate(preds).flatten(), np.concatenate(trues).flatten()


def predict_array(model, X_arr, device, batch_size=64):
  """X_arr can be 2-D (batch, 4) for MLP or 3-D (batch, T, 4) for sequence."""
  model.eval()
  X_t = torch.tensor(X_arr, dtype=torch.float32)
  preds = []
  with torch.no_grad():
    for i in range(0, len(X_t), batch_size):
      batch = X_t[i: i + batch_size].to(device)
      preds.append(model(batch).cpu().numpy())
  return np.concatenate(preds).flatten()


def permutation_importance(model, X_arr, y_true_inv, scaler_y, device,
                           n_repeats=N_PERM_REPEATS, seed=SEED):
  """Model-agnostic permutation feature importance. X_arr: (N,4) for MLP or
  (N,T,4) for sequence models."""
  rng = np.random.default_rng(seed)

  base_preds_inv = scaler_y.inverse_transform(
      predict_array(model, X_arr, device).reshape(-1, 1)).flatten()
  baseline_rmse = float(np.sqrt(np.mean((y_true_inv - base_preds_inv) ** 2)))

  importances = {}
  for f_idx, f_name in enumerate(FEATURE_NAMES):
    deltas = []
    for _ in range(n_repeats):
      X_perm = X_arr.copy()
      perm_order = rng.permutation(len(X_perm))
      if X_perm.ndim == 2:
        X_perm[:, f_idx] = X_perm[perm_order, f_idx]
      else:
        X_perm[:, :, f_idx] = X_perm[perm_order][:, :, f_idx]

      perm_preds_inv = scaler_y.inverse_transform(
          predict_array(model, X_perm, device).reshape(-1, 1)).flatten()
      perm_rmse = float(np.sqrt(np.mean((y_true_inv - perm_preds_inv) ** 2)))
      deltas.append(perm_rmse - baseline_rmse)

    deltas = np.array(deltas)
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std())
    pct_increase = (mean_delta / baseline_rmse *
                    100.0) if baseline_rmse > 0 else 0.0
    importances[f_name] = {
        'mean_delta':   mean_delta,
        'std_delta':    std_delta,
        'pct_increase': pct_increase,
    }
  return importances, baseline_rmse


def compute_metrics(y_true, y_pred, prev_price):
  """All inputs in original USD scale."""
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
  """Diebold-Mariano test with Harvey et al. (1997) small-sample correction."""
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
# SECTION 6: OUTPUT FUNCTIONS (converted to markdown per task.txt point 4)
# ============================================================

def save_metrics_report_md(all_metrics, train_summaries, data_info, split_keys):
  lines = ["# DL Model Comparison Report, WTI Crude Oil Price Forecasting"
           f" ({TAG})", ""]
  lines.append(f"Lookback window (T): {LOOKBACK} trading days.")
  lines.append(f"Train windows: {data_info['n_tr']}, "
               f"val windows: {data_info['n_val']}.")
  for sk in split_keys:
    lines.append(f"{sk.capitalize()} windows: {data_info[f'n_{sk}']} "
                 f"({data_info[f'{sk}_start']} s.d. {data_info[f'{sk}_end']}).")
  lines.append("")

  for sk in split_keys:
    lines.append(f"## {sk.capitalize()} Set Metrics, Skala Asli USD")
    lines.append("")
    lines.append("| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in MODEL_NAMES:
      m = all_metrics[name][sk]
      lines.append(
          f"| {name} | {m['MAE']:.4f} | {m['MAPE']:.4f} | {m['SMAPE']:.4f} "
          f"| {m['RMSE']:.4f} | {m['DA']:.2f} | {m['MASE']:.4f} | {m['R2']:.4f} |"
      )
    lines.append("")

  lines.append("## Training Summary")
  lines.append("")
  lines.append("| Model | Epochs | Best Val Loss | Time (s) |")
  lines.append("|---|---:|---:|---:|")
  for name in MODEL_NAMES:
    s = train_summaries[name]
    lines.append(f"| {name} | {s['epochs']} | {s['best_val']:.6f} | {s['time']:.1f} |")
  lines.append("")

  path = f"{STAT_DIR}/01_metrics_summary.md"
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + "\n")
  print(f"  [+] Saved: {path}")


def save_dm_report_md(dm_stat, dm_pval, split_name, filename):
  lines = [f"# Diebold Mariano Pairwise Test, {split_name.capitalize()} Set", "",
           "Harvey et al. (1997) small sample correction, h=1, loss adalah squared error.", "",
           "| Pair (i vs j) | DM Stat | p-value | Sig |",
           "|---|---:|---:|---|"]
  for i, m1 in enumerate(MODEL_NAMES):
    for j, m2 in enumerate(MODEL_NAMES):
      if i == j:
        continue
      lines.append(
          f"| {m1} vs {m2} | {dm_stat[i,j]:.4f} | {dm_pval[i,j]:.4f} | {sig_stars(dm_pval[i,j])} |"
      )
  lines += ["",
            "Signifikansi:",
            "",
            "- *** p<0.001",
            "- ** p<0.01",
            "- * p<0.05",
            "- ns p>=0.05",
            "",
            "Catatan, DM<0 berarti model i memiliki squared loss lebih rendah dari model j (model i lebih baik)."]
  path = f"{STAT_DIR}/{filename}"
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + "\n")
  print(f"  [+] Saved: {path}")


def save_feature_importance_report_md(fi_dict, baseline_rmse, split_name, filename):
  lines = [f"# Permutation Feature Importance, {split_name.capitalize()} Set", "",
           "Metrik, kenaikan RMSE (USD) setelah fitur diacak antar sampel.", "",
           f"Repeats: {N_PERM_REPEATS}, seed: {SEED}.", ""]
  for name in MODEL_NAMES:
    lines.append(f"## {name} (baseline RMSE = {baseline_rmse[name]:.4f} USD)")
    lines.append("")
    lines.append("| Feature | dRMSE (USD) | Std | % Increase |")
    lines.append("|---|---:|---:|---:|")
    ranked = sorted(fi_dict[name].items(),
                    key=lambda kv: kv[1]['mean_delta'], reverse=True)
    for feat_name, v in ranked:
      lines.append(
          f"| {feat_name} | {v['mean_delta']:.4f} | {v['std_delta']:.4f} | {v['pct_increase']:.2f}% |"
      )
    lines.append("")
  lines.append("Semakin tinggi dRMSE atau persen kenaikan, semakin penting fitur tersebut bagi model.")
  lines.append("Nilai negatif menandakan pengacakan fitur tidak merugikan (atau justru sedikit membantu, seperti noise) bagi prediksi model tersebut.")

  path = f"{STAT_DIR}/{filename}"
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + "\n")
  print(f"  [+] Saved: {path}")


def save_predictions_csv(dates, y_true, all_preds, split_name):
  df = pd.DataFrame({"Date": dates, "actual": y_true})
  for name in MODEL_NAMES:
    df[name] = all_preds[name]
  path = f"{STAT_DIR}/predictions_{split_name}.csv"
  df.to_csv(path, index=False)
  print(f"  [+] Saved: {path}")


# ============================================================
# SECTION 7: PLOTTING (verbatim structure from source, GRAPH_DIR per tag)
# ============================================================

def plot_feature_importance_bars(fi_dict, split_name, filename):
  fig, axes = plt.subplots(4, 2, figsize=(16, 20))
  axes = axes.flatten()
  for idx, name in enumerate(MODEL_NAMES):
    ax = axes[idx]
    ranked = sorted(fi_dict[name].items(),
                    key=lambda kv: kv[1]['pct_increase'], reverse=True)
    feat_labels = [k for k, _ in ranked]
    vals = [v['pct_increase'] for _, v in ranked]

    bars = ax.barh(feat_labels, vals, color=MODEL_COLORS[idx], edgecolor='white')
    ax.invert_yaxis()
    ax.set_title(f'{name} Feature Importance', fontsize=11, fontweight='bold')
    ax.set_xlabel('% Increase in RMSE', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    max_val = max(vals) if vals else 1.0
    offset = max_val * 0.02
    for bar, val in zip(bars, vals):
      ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
              f'{val:.2f}%', va='center', fontsize=7)

  fig.suptitle(f'Permutation Feature Importance, {split_name} Set',
               fontsize=14, fontweight='bold', y=1.005)
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_predictions(all_preds, y_true, dates, split_name, filename):
  fig, axes = plt.subplots(4, 2, figsize=(16, 20))
  axes = axes.flatten()
  for idx, name in enumerate(MODEL_NAMES):
    ax = axes[idx]
    ax.plot(dates, y_true, color='#2c3e50', linewidth=0.8, label='Actual', alpha=0.85)
    ax.plot(dates, all_preds[name], color='#e74c3c', linewidth=0.8, label='Predicted', alpha=0.75)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('WTI Price (USD)', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=30, labelsize=7)
  fig.suptitle(f'Actual vs Predicted, {split_name} Set',
               fontsize=14, fontweight='bold', y=1.005)
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_single_prediction(y_true, y_pred, dates, model_name, split_name, save_dir):
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(dates, y_true, color='#2c3e50', linewidth=0.9, label='Actual', alpha=0.85)
  ax.plot(dates, y_pred, color='#e74c3c', linewidth=0.9, label='Predicted', alpha=0.75)
  ax.set_title(f'{model_name}, Actual vs Predicted ({split_name} Set)',
               fontsize=12, fontweight='bold')
  ax.set_xlabel('Date', fontsize=10)
  ax.set_ylabel('WTI Price (USD)', fontsize=10)
  ax.legend(fontsize=9)
  ax.grid(True, linestyle='--', alpha=0.3)
  ax.tick_params(axis='x', rotation=30)
  plt.tight_layout()
  path = f"{save_dir}/{model_name}.png"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_single_feature_importance(fi_dict_for_model, model_name, split_name, save_dir, color):
  ranked = sorted(fi_dict_for_model.items(),
                  key=lambda kv: kv[1]['pct_increase'], reverse=True)
  feat_labels = [k for k, _ in ranked]
  vals = [v['pct_increase'] for _, v in ranked]

  fig, ax = plt.subplots(figsize=(9, 5.5))
  bars = ax.barh(feat_labels, vals, color=color, edgecolor='white')
  ax.invert_yaxis()
  ax.set_title(f'{model_name}, Permutation Feature Importance ({split_name} Set)',
               fontsize=12, fontweight='bold')
  ax.set_xlabel('% Increase in RMSE', fontsize=10)
  ax.tick_params(axis='y', labelsize=9)
  ax.grid(True, linestyle='--', alpha=0.3, axis='x')

  max_val = max(vals) if vals else 1.0
  offset = max_val * 0.02
  for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', fontsize=8)
  plt.tight_layout()
  path = f"{save_dir}/{model_name}.png"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_metrics_comparison(all_metrics, split_key, split_label, filename):
  metrics_keys = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'DA', 'MASE', 'R2']
  metrics_labels = ['MAE (USD)', 'MAPE (%)', 'SMAPE (%)', 'RMSE (USD)',
                    'DA (%)', 'MASE', 'R2']
  fig, axes = plt.subplots(3, 3, figsize=(18, 14))
  axes = axes.flatten()
  for idx, (key, label) in enumerate(zip(metrics_keys, metrics_labels)):
    ax = axes[idx]
    vals = [all_metrics[name][split_key][key] for name in MODEL_NAMES]
    bars = ax.bar(MODEL_NAMES, vals, color=MODEL_COLORS, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=9)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    offset = max(vals) * 0.015
    for bar, val in zip(bars, vals):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7)
  for ax in axes[len(metrics_keys):]:
    ax.set_visible(False)
  fig.suptitle(f'Metric Comparison, {split_label} Set', fontsize=14, fontweight='bold')
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_training_curves(histories, filename):
  fig, axes = plt.subplots(4, 2, figsize=(14, 18))
  axes = axes.flatten()
  for idx, name in enumerate(MODEL_NAMES):
    ax = axes[idx]
    h = histories[name]
    ep = range(1, len(h['train_loss']) + 1)
    ax.plot(ep, h['train_loss'], color='darkblue', linewidth=1.0, label='Train')
    ax.plot(ep, h['val_loss'],   color='darkorange', linewidth=1.0, label='Val')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('MSE Loss', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.3)
  fig.suptitle('Training & Validation Loss Curves', fontsize=14, fontweight='bold', y=1.005)
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


def plot_dm_heatmap(dm_stat, dm_pval, split_name, filename):
  n = len(MODEL_NAMES)
  abs_max = np.nanmax(np.abs(dm_stat[~np.eye(n, dtype=bool)]))
  vmax = abs_max + 0.5 if abs_max > 0 else 1.0

  fig, ax = plt.subplots(figsize=(9, 7))
  im = ax.imshow(dm_stat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
  plt.colorbar(im, ax=ax, label='DM Statistic')

  ax.set_xticks(range(n))
  ax.set_yticks(range(n))
  ax.set_xticklabels(MODEL_NAMES, rotation=30, ha='right', fontsize=9)
  ax.set_yticklabels(MODEL_NAMES, fontsize=9)
  ax.set_xlabel('Model j (baseline)', fontsize=10)
  ax.set_ylabel('Model i (challenger)', fontsize=10)

  for i in range(n):
    for j in range(n):
      if i == j:
        txt = '-'
        col = 'black'
      else:
        txt = f"{dm_stat[i,j]:.2f}\n{sig_stars(dm_pval[i,j])}"
        col = 'white' if abs(dm_stat[i, j]) > vmax * 0.55 else 'black'
      ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=col)

  ax.set_title(
      f'DM Test Heatmap, {split_name} Set\n'
      '(Row i vs Col j, negatif berarti i memiliki loss lebih rendah)',
      fontsize=11, fontweight='bold'
  )
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


# ============================================================
# SECTION 8: MAIN ORCHESTRATION
# ============================================================

def main():
  os.makedirs(STAT_DIR, exist_ok=True)
  log_path = f"{STAT_DIR}/00_training_log.txt"
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
    print(f"[+] Full training log saved: {log_path}")


def _run_pipeline():
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("=" * 66)
  print(f"  STAGE 05: DL MODEL TRAINING & COMPARISON, run={TAG}")
  print(f"  Device: {device} | Seed: {SEED} | Lookback T: {LOOKBACK}")
  print("=" * 66)

  eval_dirs = [STAT_DIR, GRAPH_DIR, MODEL_DIR, PRED_TEST_DIR, FI_TEST_DIR]
  print("\n[*] Loading data...")
  splits, scaler_X, scaler_y, has_unseen = load_data()
  split_keys = [k for k in ["test", "unseen"] if k in splits]
  if has_unseen:
    eval_dirs += [PRED_UNSEEN_DIR, FI_UNSEEN_DIR]
  for d in eval_dirs:
    os.makedirs(d, exist_ok=True)

  X_train_full, y_train_full, dates_train_full = splits["train"]

  # ----------------------------------------------------------
  # Train/val split (chronological tail of train as validation)
  # ----------------------------------------------------------
  n_val = int(len(X_train_full) * VAL_FRACTION)
  n_tr = len(X_train_full) - n_val

  X_tr_seq, y_tr_seq = X_train_full[:n_tr],  y_train_full[:n_tr]
  X_val_seq, y_val_seq = X_train_full[n_tr:], y_train_full[n_tr:]

  # MLP uses only the most-recent timestep (t-0, i.e. lag_1) of each window
  X_tr_flat = X_tr_seq[:, -1, :]
  X_val_flat = X_val_seq[:, -1, :]

  print(f"    Train windows  : {n_tr}")
  print(f"    Val windows    : {n_val}")
  for sk in split_keys:
    print(f"    {sk.capitalize()} windows : {len(splits[sk][1])}")

  seq_tr_ld = make_loader(X_tr_seq,  y_tr_seq,  BATCH_SIZE, shuffle=True)
  seq_val_ld = make_loader(X_val_seq, y_val_seq, BATCH_SIZE)
  mlp_tr_ld = make_loader(X_tr_flat, y_tr_seq,  BATCH_SIZE, shuffle=True)
  mlp_val_ld = make_loader(X_val_flat, y_val_seq, BATCH_SIZE)

  eval_loaders = {}   # eval_loaders[split][kind] -> loader, kind in {'mlp','seq'}
  for sk in split_keys:
    X_e, y_e, _ = splits[sk]
    eval_loaders[sk] = {
        'mlp': make_loader(X_e[:, -1, :], y_e, BATCH_SIZE),
        'seq': make_loader(X_e,           y_e, BATCH_SIZE),
    }

  model_registry = {
      'MLP':         (MLPModel(),         mlp_tr_ld, mlp_val_ld, 'mlp'),
      'RNN':         (RNNModel(),         seq_tr_ld, seq_val_ld, 'seq'),
      'LSTM':        (LSTMModel(),        seq_tr_ld, seq_val_ld, 'seq'),
      'BiLSTM':      (BiLSTMModel(),      seq_tr_ld, seq_val_ld, 'seq'),
      'GRU':         (GRUModel(),         seq_tr_ld, seq_val_ld, 'seq'),
      'TCN':         (TCNModel(),         seq_tr_ld, seq_val_ld, 'seq'),
      'Transformer': (TransformerModel(), seq_tr_ld, seq_val_ld, 'seq'),
      'Informer':    (InformerModel(),    seq_tr_ld, seq_val_ld, 'seq'),
  }

  # ----------------------------------------------------------
  # Train all models
  # ----------------------------------------------------------
  trained_models = {}
  histories = {}
  train_summaries = {}

  for name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"[*] Training {name}...")
    model, tr_ld, val_ld, _ = model_registry[name]
    trained_model, hist, elapsed = train_model(model, tr_ld, val_ld, name, device)
    trained_models[name] = trained_model
    histories[name] = hist
    train_summaries[name] = {
        'epochs':   len(hist['train_loss']),
        'best_val': min(hist['val_loss']),
        'time':     elapsed,
    }
    save_path = f"{MODEL_DIR}/{name.lower()}_model.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"  [+] Model saved: {save_path}")

  # ----------------------------------------------------------
  # Predictions & metrics
  # ----------------------------------------------------------
  print("\n[*] Generating predictions and computing metrics...")

  def inv_y(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

  def prev_price_inv(X_last_timestep):
    """Inverse-transform Trend component (col 0, t-0) via scaler_X, used as
    previous-day reference for directional accuracy/MASE naive forecast."""
    dummy = np.zeros((len(X_last_timestep), N_FEATURES), dtype=np.float32)
    dummy[:, 0] = X_last_timestep[:, 0]
    return scaler_X.inverse_transform(dummy)[:, 0]

  all_metrics = {name: {} for name in MODEL_NAMES}
  all_preds = {sk: {} for sk in split_keys}
  errors = {sk: {} for sk in split_keys}
  y_true_inv = {}
  prev_price = {}

  for sk in split_keys:
    _, y_e, _ = splits[sk]
    y_true_inv[sk] = inv_y(y_e)
    prev_price[sk] = prev_price_inv(splits[sk][0][:, -1, :])

  for name in MODEL_NAMES:
    _, _, _, kind = model_registry[name]
    mdl = trained_models[name]
    for sk in split_keys:
      loader = eval_loaders[sk][kind]
      p_sc, _ = predict(mdl, loader, device)
      p_inv = inv_y(p_sc)
      all_preds[sk][name] = p_inv
      errors[sk][name] = y_true_inv[sk] - p_inv
      all_metrics[name][sk] = compute_metrics(y_true_inv[sk], p_inv, prev_price[sk])
    parts = " | ".join(
        f"{sk.upper()}: MAE={all_metrics[name][sk]['MAE']:.4f} "
        f"MAPE={all_metrics[name][sk]['MAPE']:.4f}% R2={all_metrics[name][sk]['R2']:.4f}"
        for sk in split_keys
    )
    print(f"  {name:<12} {parts}")

  data_info = {'n_tr': n_tr, 'n_val': n_val}
  for sk in split_keys:
    dates_e = splits[sk][2]
    data_info[f'n_{sk}'] = len(dates_e)
    data_info[f'{sk}_start'] = dates_e.min().strftime('%Y-%m-%d')
    data_info[f'{sk}_end'] = dates_e.max().strftime('%Y-%m-%d')

  save_metrics_report_md(all_metrics, train_summaries, data_info, split_keys)

  for sk in split_keys:
    save_predictions_csv(splits[sk][2].strftime('%Y-%m-%d'), y_true_inv[sk], all_preds[sk], sk)

  # ----------------------------------------------------------
  # Diebold-Mariano pairwise tests
  # ----------------------------------------------------------
  print(f"\n[*] Running Diebold-Mariano pairwise tests (28 pairs x {len(split_keys)} splits)...")
  n = len(MODEL_NAMES)
  dm_stat = {sk: np.zeros((n, n)) for sk in split_keys}
  dm_pval = {sk: np.ones((n, n)) for sk in split_keys}

  for sk in split_keys:
    for i, m1 in enumerate(MODEL_NAMES):
      for j, m2 in enumerate(MODEL_NAMES):
        if i == j:
          continue
        s, p = diebold_mariano_test(errors[sk][m1], errors[sk][m2])
        dm_stat[sk][i, j] = s
        dm_pval[sk][i, j] = p

  dm_filenames = {"test": "02_dm_test.md", "unseen": "02b_dm_unseen.md"}
  for sk in split_keys:
    save_dm_report_md(dm_stat[sk], dm_pval[sk], sk, dm_filenames[sk])
    plot_dm_heatmap(dm_stat[sk], dm_pval[sk], sk.capitalize(), f"dm_heatmap_{sk}.png")

  # ----------------------------------------------------------
  # Permutation feature importance
  # ----------------------------------------------------------
  print(f"\n[*] Computing permutation feature importance "
        f"({N_PERM_REPEATS} repeats x {len(FEATURE_NAMES)} features x "
        f"{len(MODEL_NAMES)} models x {len(split_keys)} splits)...")

  fi_dicts = {sk: {} for sk in split_keys}
  fi_baselines = {sk: {} for sk in split_keys}
  for name in MODEL_NAMES:
    _, _, _, kind = model_registry[name]
    mdl = trained_models[name]
    for sk in split_keys:
      X_e = splits[sk][0][:, -1, :] if kind == 'mlp' else splits[sk][0]
      fi, base_rmse = permutation_importance(mdl, X_e, y_true_inv[sk], scaler_y, device)
      fi_dicts[sk][name] = fi
      fi_baselines[sk][name] = base_rmse

  fi_report_filenames = {"test": "03_feature_importance_test.md",
                         "unseen": "03b_feature_importance_unseen.md"}
  fi_plot_filenames = {"test": "feature_importance_all_models_test.png",
                       "unseen": "feature_importance_all_models_unseen.png"}
  fi_single_dirs = {"test": FI_TEST_DIR, "unseen": FI_UNSEEN_DIR if has_unseen else None}
  for sk in split_keys:
    save_feature_importance_report_md(fi_dicts[sk], fi_baselines[sk], sk, fi_report_filenames[sk])
    plot_feature_importance_bars(fi_dicts[sk], sk.capitalize(), fi_plot_filenames[sk])
    for idx, name in enumerate(MODEL_NAMES):
      plot_single_feature_importance(fi_dicts[sk][name], name, sk.capitalize(),
                                     fi_single_dirs[sk], MODEL_COLORS[idx])

  # ----------------------------------------------------------
  # Prediction & metric comparison plots
  # ----------------------------------------------------------
  print("\n[*] Generating plots...")
  pred_plot_filenames = {"test": "actual_vs_predicted_test.png",
                         "unseen": "actual_vs_predicted_unseen.png"}
  pred_single_dirs = {"test": PRED_TEST_DIR, "unseen": PRED_UNSEEN_DIR if has_unseen else None}
  metrics_plot_filenames = {"test": "metrics_comparison_test.png",
                            "unseen": "metrics_comparison_unseen.png"}
  for sk in split_keys:
    dates_e = splits[sk][2]
    plot_predictions(all_preds[sk], y_true_inv[sk], dates_e, sk.capitalize(), pred_plot_filenames[sk])
    for name in MODEL_NAMES:
      plot_single_prediction(y_true_inv[sk], all_preds[sk][name], dates_e, name,
                             sk.capitalize(), pred_single_dirs[sk])
    plot_metrics_comparison(all_metrics, sk, sk.capitalize(), metrics_plot_filenames[sk])

  plot_training_curves(histories, "training_curves.png")

  print(f"\n[+] Done. Run tag: {TAG}. Outputs in {STAT_DIR} and {GRAPH_DIR}.")


if __name__ == "__main__":
  main()
