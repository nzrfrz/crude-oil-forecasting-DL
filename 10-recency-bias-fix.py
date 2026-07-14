# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
# STAGE 10: TRANSFORMER + INFORMER RECENCY-BIAS MITIGATION
# WTI Crude Oil Price Forecasting, CEEMDAN 4-component input (Final Task DL)
#
# Stage 09 XAI found both attention models (Transformer, Informer) have the
# lowest SHAP timestep concentration and the lowest t-0 (most-recent-day)
# attribution share among all sequence models, in every split (full test,
# full unseen, wu test) -- see evaluations/xai/statistical/{tag}/03_concentration_metrics.csv
# and 02_timestep_attribution_*.csv. This correlates with them having the
# worst MAE among sequence models on the full-run test set.
#
# This script does NOT touch 05-dl-model-training.py. It defines "Fixed"
# variants of Transformer and Informer with two targeted changes:
#   1. Learned positional encoding (nn.Parameter) instead of fixed sinusoidal
#      -- lets gradient descent shape position vectors directly, including a
#      recency skew, instead of relying on a fixed pattern with no incentive
#      to prefer any particular position.
#   2. Last-timestep pooling (x[:, -1, :]) instead of mean pooling over the
#      whole sequence -- the same convention every recurrent/TCN model in
#      this project already uses. Mean pooling weights every lookback day
#      equally by construction, which structurally caps how concentrated
#      attribution on t-0 can ever get, regardless of what attention learns.
#
# The original (baseline) checkpoints from Stage 05 are loaded as-is for
# comparison, NOT retrained here. Only the Fixed variants are trained from
# scratch, with the same training procedure as Stage 05. Both are then
# re-explained with the same SHAP timestep-attribution method from Stage 09
# to check whether concentration actually improved.
#
# Adaptasi dari D:\Coding\#bigdata\crude-oil-forecasting-DL\10-transformer-recency-fix.py,
# diperluas ke Informer (bukan cuma Transformer) karena XAI kita menunjukkan
# Informer punya kelemahan pooling yang sama persis (x.mean(dim=1)), dan
# dijalankan untuk kedua run (full, wu).
# ==================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import os
import sys
import argparse
import math
import time
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

import shap
from scipy.stats import entropy as scipy_entropy

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
# SECTION 1: CONFIGURATION (mirrors Stage 05 exactly, for a fair comparison)
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("--run", choices=["full", "wu"], required=True)
ARGS = ap.parse_args()
TAG = ARGS.run

SEED = 42
LOOKBACK = 10
N_FEATURES = 4
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-5
VAL_FRACTION = 0.10
DPI = 300

# XAI re-check settings (mirrors Stage 09)
XAI_EVAL_SAMPLE_SIZE = 200
SHAP_BACKGROUND_SIZE = 200
SHAP_NSAMPLES = 50

# ProbSparseAttention samples candidate keys with torch.randint, so Informer's
# forward pass is stochastic even in eval mode; attributions are averaged
# over repeated draws for a stable estimate (see Stage 09).
STABILITY_REPEATS = {"Transformer": 1, "Informer": 10}

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
MODEL_DIR = f"models/{TAG}"
STAT_DIR = f"evaluations/statistical/recency-fix/{TAG}"
GRAPH_DIR = f"evaluations/graphical/recency-fix/{TAG}"

MODEL_NAMES = ["Transformer", "Informer"]
MODEL_COLORS = {"Baseline": "#2C7FB8", "Fixed": "#DA8BC3"}


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
# SECTION 3: BASELINE MODEL DEFINITIONS
# Verbatim from 05-dl-model-training.py, needed only to reconstruct the
# exact module structure so the Stage 05 checkpoints load correctly.
# ============================================================

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
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerModelBaseline(nn.Module):
  """Exact copy of Stage 05's TransformerModel (sinusoidal PE + mean
  pooling). Only used to load the existing Stage 05 checkpoint."""

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
  """ProbSparse self-attention from Informer (Zhou et al. 2021)."""

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


class InformerModelBaseline(nn.Module):
  """Exact copy of Stage 05's InformerModel (sinusoidal PE + mean pooling).
  Only used to load the existing Stage 05 checkpoint."""

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
# SECTION 4: FIXED MODEL DEFINITIONS (learned PE + last-timestep pooling)
# ============================================================

class LearnedPositionalEncoding(nn.Module):
  """Learned (trainable) positional embedding, one vector per lookback
  slot, instead of a fixed sinusoidal pattern. Lets gradient descent shape
  the position vectors directly, including a recency skew, rather than
  relying on a fixed encoding the model has no incentive to weight unevenly."""

  def __init__(self, d_model, max_len, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
    nn.init.trunc_normal_(self.pos_embed, std=0.02)

  def forward(self, x):
    return self.dropout(x + self.pos_embed[:, :x.size(1), :])


class TransformerModelFixed(nn.Module):
  """Encoder-only Transformer with learned positional encoding and
  last-timestep pooling instead of Stage 05's sinusoidal PE + mean pooling.
  Same d_model/nhead/layer count as the baseline for a fair comparison."""

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
  """Informer encoder-only model with learned positional encoding and
  last-timestep pooling instead of Stage 05's sinusoidal PE + mean pooling.
  ProbSparse attention itself is unchanged."""

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


MODEL_SPECS = {
    "Transformer": (TransformerModelBaseline, TransformerModelFixed),
    "Informer":    (InformerModelBaseline,    InformerModelFixed),
}


# ============================================================
# SECTION 5: TRAINING (mirrors Stage 05's train_model/EarlyStopping exactly)
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


def train_model(model, train_loader, val_loader, name, device):
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=LR_FACTOR,
      patience=LR_PATIENCE, min_lr=LR_MIN
  )
  criterion = nn.MSELoss()
  stopper = EarlyStopping(patience=PATIENCE)
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

    if epoch == 1 or epoch % 20 == 0:
      lr_now = optimizer.param_groups[0]['lr']
      print(f"  [{name}] Epoch {epoch:3d} | Train={tr_loss:.6f} | "
            f"Val={va_loss:.6f} | LR={lr_now:.2e}")

    if stopper.step(va_loss, model):
      print(f"  [{name}] Early stop @ epoch {epoch} (best_val={stopper.best_loss:.6f})")
      break

  stopper.restore_best(model)
  elapsed = time.time() - t0
  print(f"  [{name}] Finished in {elapsed:.1f}s | Best val: {stopper.best_loss:.6f}")
  return model, elapsed


# ============================================================
# SECTION 6: METRICS (mirrors Stage 05's compute_metrics exactly)
# ============================================================

def predict_array(model, X_arr, device, batch_size=64):
  model.eval()
  X_t = torch.tensor(X_arr, dtype=torch.float32)
  preds = []
  with torch.no_grad():
    for i in range(0, len(X_t), batch_size):
      batch = X_t[i: i + batch_size].to(device)
      preds.append(model(batch).cpu().numpy())
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
  """Inverse-transform Trend component (col 0, t-0) via scaler_X, used as
  previous-day reference for directional accuracy/MASE naive forecast
  (mirrors Stage 05's prev_price_inv)."""
  dummy = np.zeros((len(X_last_timestep), N_FEATURES), dtype=np.float32)
  dummy[:, 0] = X_last_timestep[:, 0]
  return scaler_X.inverse_transform(dummy)[:, 0]


# ============================================================
# SECTION 7: XAI RE-CHECK (mirrors Stage 09's SHAP timestep attribution)
# ============================================================

def subsample(X_arr, n, seed=SEED):
  if len(X_arr) <= n:
    return X_arr
  rng = np.random.default_rng(seed)
  idx = rng.choice(len(X_arr), size=n, replace=False)
  return X_arr[idx]


def compute_shap_timestep_attribution(model, X_bg, X_eval, device, n_repeats, seed=SEED):
  """SHAP GradientExplainer attributions -> mean |attr| per timestep, summed
  over features, averaged over `n_repeats` draws (needed for Informer's
  stochastic ProbSparse attention)."""
  bg_t = torch.tensor(X_bg,   dtype=torch.float32, device=device)
  eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

  runs = []
  for r in range(n_repeats):
    torch.manual_seed(seed + r)
    explainer = shap.GradientExplainer(model, bg_t)
    sv = explainer.shap_values(eval_t, nsamples=SHAP_NSAMPLES)
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    if sv.ndim == eval_t.dim() + 1 and sv.shape[-1] == 1:
      sv = sv[..., 0]
    runs.append(sv)
  sv_mean = np.mean(runs, axis=0)
  abs_attr = np.abs(sv_mean)
  return abs_attr.sum(axis=2).mean(axis=0)


def concentration_score(v):
  v = np.asarray(v, dtype=np.float64).reshape(-1)
  if v.sum() <= 0 or len(v) <= 1:
    return 0.0
  p = v / v.sum()
  h = scipy_entropy(p, base=2)
  h_max = math.log2(len(v))
  return float(1.0 - (h / h_max if h_max > 0 else 0.0))


# ============================================================
# SECTION 8: REPORTS
# ============================================================

def save_summary_report(results, split_keys, filename):
  lines = [f"# Recency-Bias Mitigation, Baseline vs Fixed ({TAG})", "",
           "Fixed, learned positional encoding plus last timestep pooling.",
           "Baseline, checkpoint Stage 05 asli, sinusoidal PE plus mean pooling, tidak dilatih ulang.",
           ""]
  for name in MODEL_NAMES:
    r = results[name]
    lines.append(f"## {name}")
    lines.append("")
    for sk in split_keys:
      lines.append(f"### {sk.capitalize()} Set Metrics")
      lines.append("")
      lines.append("| Varian | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |")
      lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
      for label, m in [("Baseline", r['metrics_base'][sk]), ("Fixed", r['metrics_fixed'][sk])]:
        lines.append(
            f"| {label} | {m['MAE']:.4f} | {m['MAPE']:.4f} | {m['SMAPE']:.4f} "
            f"| {m['RMSE']:.4f} | {m['DA']:.2f} | {m['MASE']:.4f} | {m['R2']:.4f} |"
        )
      lines.append("")
    lines.append("### Timestep Attribution Concentration Score (test set, SHAP)")
    lines.append("")
    lines.append(f"- Baseline: {r['conc_base']:.4f}")
    lines.append(f"- Fixed: {r['conc_fixed']:.4f}")
    lines.append("")
    lines.append("Skor 0 berarti atribusi menyebar rata ke semua 10 hari lookback, "
                  "skor 1 berarti terkonsentrasi penuh di satu hari.")
    lines.append("")
    verdict = "MENINGKAT" if r['conc_fixed'] > r['conc_base'] else "TIDAK MENINGKAT"
    lines.append(f"Verdict, konsentrasi recency {verdict} "
                 f"({r['conc_base']:.4f} menjadi {r['conc_fixed']:.4f}).")
    lines.append("")

  path = f"{STAT_DIR}/{filename}"
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + "\n")
  print(f"  [+] Saved: {path}")


def plot_timestep_comparison(ts_base, ts_fixed, name, filename):
  lags = [f"t-{LOOKBACK - 1 - t}" for t in range(LOOKBACK)]
  x = np.arange(LOOKBACK)
  fig, ax = plt.subplots(figsize=(9, 5))
  ax.plot(x, ts_base,  marker='o', label='Baseline (mean pool)', color=MODEL_COLORS["Baseline"])
  ax.plot(x, ts_fixed, marker='s', label='Fixed (last-timestep pool)', color=MODEL_COLORS["Fixed"])
  ax.set_xticks(x)
  ax.set_xticklabels(lags)
  ax.set_xlabel('Lookback timestep (t-9 = oldest, t-0 = terbaru)')
  ax.set_ylabel('Mean |SHAP| (dijumlah antar fitur)')
  ax.set_title(f'{name} Timestep Attribution, Baseline vs Fixed')
  ax.legend()
  fig.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  print(f"  [+] Saved: {path}")


def plot_metrics_comparison(metrics_base, metrics_fixed, name, split, filename):
  keys = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'DA', 'MASE', 'R2']
  labels = ['MAE (USD)', 'MAPE (%)', 'SMAPE (%)', 'RMSE (USD)', 'DA (%)', 'MASE', 'R2']
  fig, axes = plt.subplots(3, 3, figsize=(15, 12))
  axes = axes.flatten()
  for idx, (key, label) in enumerate(zip(keys, labels)):
    ax = axes[idx]
    vals = [metrics_base[split][key], metrics_fixed[split][key]]
    bars = ax.bar(['Baseline', 'Fixed'], vals,
                  color=[MODEL_COLORS["Baseline"], MODEL_COLORS["Fixed"]], edgecolor='white')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    offset = max(abs(v) for v in vals) * 0.02 if any(vals) else 0.01
    for bar, val in zip(bars, vals):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
              f'{val:.3f}', ha='center', va='bottom', fontsize=8)
  for ax in axes[len(keys):]:
    ax.set_visible(False)
  fig.suptitle(f'{name}, Baseline vs Fixed, {split.capitalize()} Set',
               fontsize=14, fontweight='bold')
  plt.tight_layout()
  path = f"{GRAPH_DIR}/{filename}"
  plt.savefig(path, dpi=DPI, bbox_inches='tight')
  plt.close()
  print(f"  [+] Saved: {path}")


# ============================================================
# SECTION 9: PIPELINE
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

  print("=" * 70)
  print(f"  STAGE 10: RECENCY-BIAS MITIGATION, run={TAG}")
  print(f"  Device: {device} | Seed: {SEED} | Lookback T: {LOOKBACK}")
  print("=" * 70)

  for d in [STAT_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

  print("\n[*] Loading data...")
  splits, scaler_X, scaler_y = load_data()
  split_keys = [k for k in ["test", "unseen"] if k in splits]

  X_train_full, y_train_full = splits["train"]
  n_val = int(len(X_train_full) * VAL_FRACTION)
  n_tr = len(X_train_full) - n_val
  X_tr_seq, y_tr_seq = X_train_full[:n_tr],  y_train_full[:n_tr]
  X_val_seq, y_val_seq = X_train_full[n_tr:], y_train_full[n_tr:]

  train_ld = make_loader(X_tr_seq,  y_tr_seq,  BATCH_SIZE, shuffle=True)
  val_ld = make_loader(X_val_seq, y_val_seq, BATCH_SIZE)

  def inv_y(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

  y_true_inv = {sk: inv_y(splits[sk][1]) for sk in split_keys}
  prev_price = {sk: prev_price_inv(splits[sk][0][:, -1, :], scaler_X) for sk in split_keys}

  results = {}
  for name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"[*] {name}")
    print(f"{'='*60}")
    BaselineClass, FixedClass = MODEL_SPECS[name]

    print(f"  [*] Loading baseline {name} checkpoint (Stage 05, unchanged)...")
    baseline_model = BaselineClass().to(device)
    baseline_model.load_state_dict(
        torch.load(f"{MODEL_DIR}/{name.lower()}_model.pt", map_location=device))
    baseline_model.eval()

    print(f"  [*] Training Fixed {name} (learned PE + last-timestep pooling)...")
    fixed_model = FixedClass()
    fixed_model, elapsed = train_model(fixed_model, train_ld, val_ld, f"{name}-Fixed", device)
    fixed_path = f"{MODEL_DIR}/{name.lower()}_recency_fixed_model.pt"
    torch.save(fixed_model.state_dict(), fixed_path)
    print(f"  [+] Saved: {fixed_path}")

    metrics_base, metrics_fixed = {}, {}
    for label, model, metrics_dict in [("baseline", baseline_model, metrics_base),
                                       ("fixed", fixed_model, metrics_fixed)]:
      for sk in split_keys:
        p_inv = scaler_y.inverse_transform(
            predict_array(model, splits[sk][0], device).reshape(-1, 1)).flatten()
        metrics_dict[sk] = compute_metrics(y_true_inv[sk], p_inv, prev_price[sk])
      parts = " | ".join(
          f"{sk.upper()}: MAE={metrics_dict[sk]['MAE']:.4f} MAPE={metrics_dict[sk]['MAPE']:.4f}%"
          for sk in split_keys
      )
      print(f"    [{label}] {parts}")

    print(f"  [*] Re-running SHAP timestep attribution on test set...")
    X_eval_s = subsample(splits["test"][0], XAI_EVAL_SAMPLE_SIZE)
    X_bg = subsample(X_tr_seq, SHAP_BACKGROUND_SIZE)
    repeats = STABILITY_REPEATS[name]

    ts_base = compute_shap_timestep_attribution(baseline_model, X_bg, X_eval_s, device, repeats)
    ts_fixed = compute_shap_timestep_attribution(fixed_model,    X_bg, X_eval_s, device, repeats)
    conc_base = concentration_score(ts_base)
    conc_fixed = concentration_score(ts_fixed)
    print(f"    Baseline timestep concentration: {conc_base:.4f}")
    print(f"    Fixed    timestep concentration: {conc_fixed:.4f}")

    results[name] = {
        'metrics_base': metrics_base, 'metrics_fixed': metrics_fixed,
        'ts_base': ts_base, 'ts_fixed': ts_fixed,
        'conc_base': conc_base, 'conc_fixed': conc_fixed,
    }

  print("\n[*] Saving reports...")
  save_summary_report(results, split_keys, "01_recency_fix_summary.md")
  for name in MODEL_NAMES:
    r = results[name]
    plot_timestep_comparison(r['ts_base'], r['ts_fixed'], name,
                             f"02_timestep_attribution_comparison_{name.lower()}.png")
    for sk in split_keys:
      plot_metrics_comparison(r['metrics_base'], r['metrics_fixed'], name, sk,
                              f"03_metrics_comparison_{sk}_{name.lower()}.png")

  print(f"\n[+] Done. Run tag: {TAG}. Outputs in {STAT_DIR} and {GRAPH_DIR}.")


if __name__ == "__main__":
  main()
