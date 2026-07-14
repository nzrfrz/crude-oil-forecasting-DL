# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
# STAGE 09: XAI EXPLAINABILITY (8 ARCHITECTURES)
# WTI Crude Oil Price Forecasting, CEEMDAN 4-component input (Final Task DL)
# Explains the 8 trained architectures from Stage 05 using two gradient-based
# attribution methods:
#   - SHAP (shap.GradientExplainer)   -> expected-gradient attributions
#   - Integrated Gradients (captum)   -> path-integral attributions
#
# Attributions are aggregated per CEEMDAN component and per lookback
# timestep, then a concentration score (1 - normalized Shannon entropy) is
# correlated (Spearman) against forecast error, to check whether models that
# concentrate attribution more sharply achieve lower MAE.
#
# Adaptasi dari D:\Coding\#bigdata\crude-oil-forecasting-DL\09-xai-explainability.py.
# Kelas model dan fungsi atribusi inti disalin verbatim dari 05-dl-model-training.py
# (N_FEATURES=4, sudah pre-windowed dari 04-fe-and-split.py), jadi tidak perlu
# create_sequences/create_eval_sequences seperti sumber. MLP memakai input flat
# t-0 (lag_1, timestep terakhir dari window), sama seperti kesepakatan Stage 05.
# ==================================================

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import os
import sys
import argparse
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from scipy.stats import entropy as scipy_entropy, spearmanr

import shap
from captum.attr import IntegratedGradients

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
DPI = 300

# Cap on how many eval windows get explained per model (SHAP/IG cost scales
# with eval-set size x nsamples/n_steps). Test/unseen splits here run into
# the hundreds-to-low-thousands of windows, well above what full SHAP+IG
# needs to give a stable estimate, so a random subsample keeps runtime bounded.
XAI_EVAL_SAMPLE_SIZE = 200
SHAP_BACKGROUND_SIZE = 200
SHAP_NSAMPLES = 50
IG_STEPS = 50

# Informer's ProbSparseAttention samples candidate keys with torch.randint,
# so each forward pass is stochastic even in eval mode. Attributions are
# averaged over repeated draws to get a stable estimate; other models are
# fully deterministic in eval mode, so 1 repeat is exact and sufficient.
STABILITY_REPEATS = {
    "MLP": 1, "RNN": 1, "LSTM": 1, "BiLSTM": 1, "GRU": 1, "TCN": 1,
    "Transformer": 1, "Informer": 10,
}

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
MODEL_DIR = f"models/{TAG}"
STAT_DIR = f"evaluations/xai/statistical/{TAG}"
GRAPH_DIR = f"evaluations/xai/graphical/{TAG}"

FEATURE_ATTR_DIR = f"{GRAPH_DIR}/feature-attribution"
TIMESTEP_ATTR_DIR = f"{GRAPH_DIR}/timestep-attribution"
HEATMAP_DIR = f"{GRAPH_DIR}/attribution-heatmaps"
COMPARISON_DIR = f"{GRAPH_DIR}/comparison"

MODEL_NAMES = ["MLP", "RNN", "LSTM", "BiLSTM",
               "GRU", "TCN", "Transformer", "Informer"]
SEQUENCE_MODELS = ["RNN", "LSTM", "BiLSTM",
                    "GRU", "TCN", "Transformer", "Informer"]

FEATURE_NAMES = ["Trend", "IMF_Group1", "IMF_Group2", "Residual"]

MODEL_COLORS = {
    "MLP": "#4C72B0", "RNN": "#937860", "LSTM": "#DD8452", "BiLSTM": "#55A868",
    "GRU": "#C44E52", "TCN": "#8172B3", "Transformer": "#2C7FB8", "Informer": "#E69F00",
}

CHECKPOINTS = {name: f"{MODEL_DIR}/{name.lower()}_model.pt" for name in MODEL_NAMES}


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
  return splits, scaler_X, scaler_y, has_unseen


# ============================================================
# SECTION 3: MODEL DEFINITIONS (verbatim from 05-dl-model-training.py)
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
  """Vanilla RNN (tanh nonlinearity)."""

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
  """Causal (left-only) 1D convolution via symmetric padding + right trim."""

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
  """Temporal Convolutional Network with dilated causal convolutions."""

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


class TransformerModel(nn.Module):
  """Encoder-only Transformer with mean pooling over sequence dim."""

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


class InformerModel(nn.Module):
  """Informer encoder-only model with ProbSparse attention."""

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


MODEL_CLASSES = {
    "MLP": MLPModel, "RNN": RNNModel, "LSTM": LSTMModel, "BiLSTM": BiLSTMModel,
    "GRU": GRUModel, "TCN": TCNModel, "Transformer": TransformerModel,
    "Informer": InformerModel,
}


def load_trained_model(name, device):
  model = MODEL_CLASSES[name]()
  state = torch.load(CHECKPOINTS[name], map_location=device)
  model.load_state_dict(state)
  return model.to(device).eval()


# ============================================================
# SECTION 4: PREDICTION & METRICS
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


def compute_metrics(y_true, y_pred):
  mae = float(np.mean(np.abs(y_true - y_pred)))
  rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
  return {'MAE': mae, 'RMSE': rmse}


# ============================================================
# SECTION 5: ATTRIBUTION COMPUTATION (SHAP + Integrated Gradients)
# ============================================================

def compute_shap_attributions(model, X_bg, X_eval, device, n_repeats, seed=SEED):
  """SHAP GradientExplainer attributions, averaged over `n_repeats` draws
  to stabilize models with stochastic forward passes (Informer)."""
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
  return np.mean(runs, axis=0)


def compute_ig_attributions(model, X_eval, device, n_repeats, seed=SEED, batch_size=64):
  """Integrated Gradients attributions (zero baseline), averaged over
  `n_repeats` draws to stabilize models with stochastic forward passes."""
  ig = IntegratedGradients(model)
  eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)

  runs = []
  for r in range(n_repeats):
    torch.manual_seed(seed + r)
    attrs = []
    for i in range(0, len(eval_t), batch_size):
      batch = eval_t[i: i + batch_size]
      baseline = torch.zeros_like(batch)
      a = ig.attribute(batch, baselines=baseline, n_steps=IG_STEPS)
      attrs.append(a.detach().cpu().numpy())
    runs.append(np.concatenate(attrs, axis=0))
  return np.mean(runs, axis=0)


def aggregate_feature_importance(attr_arr):
  """(N,4) or (N,T,4) -> (4,) mean absolute attribution per feature,
  summed over timesteps first for sequence models."""
  abs_attr = np.abs(attr_arr)
  if abs_attr.ndim == 3:
    abs_attr = abs_attr.sum(axis=1)
  return abs_attr.mean(axis=0)


def aggregate_timestep_importance(attr_arr):
  """(N,T,4) -> (T,) mean absolute attribution per lookback timestep,
  summed over features. Returns None for non-sequence (MLP) input."""
  if attr_arr.ndim != 3:
    return None
  abs_attr = np.abs(attr_arr)
  return abs_attr.sum(axis=2).mean(axis=0)


def concentration_score(v):
  """1 - normalized Shannon entropy of a non-negative importance vector.
  0 = uniformly spread across all entries, 1 = fully concentrated on one."""
  v = np.asarray(v, dtype=np.float64).reshape(-1)
  if v.sum() <= 0 or len(v) <= 1:
    return 0.0
  p = v / v.sum()
  h = scipy_entropy(p, base=2)
  h_max = math.log2(len(v))
  return float(1.0 - (h / h_max if h_max > 0 else 0.0))


# ============================================================
# SECTION 6: STATISTICAL REPORTS
# ============================================================

def save_feature_attribution_report(feature_shap, feature_ig, filename):
  rows = []
  for name in MODEL_NAMES:
    row = {'Model': name}
    for i, f in enumerate(FEATURE_NAMES):
      row[f'SHAP_{f}'] = feature_shap[name][i]
      row[f'IG_{f}'] = feature_ig[name][i]
    rows.append(row)
  pd.DataFrame(rows).to_csv(filename, index=False)
  print(f"  [+] Saved: {filename}")


def save_timestep_attribution_report(timestep_shap, timestep_ig, filename):
  rows = []
  for name in SEQUENCE_MODELS:
    row = {'Model': name}
    for t in range(LOOKBACK):
      row[f'SHAP_t-{LOOKBACK - 1 - t}'] = timestep_shap[name][t]
      row[f'IG_t-{LOOKBACK - 1 - t}'] = timestep_ig[name][t]
    rows.append(row)
  pd.DataFrame(rows).to_csv(filename, index=False)
  print(f"  [+] Saved: {filename}")


def save_concentration_report(records, filename):
  pd.DataFrame(records).to_csv(filename, index=False)
  print(f"  [+] Saved: {filename}")


def save_correlation_report(records, filename):
  with open(filename, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("CONCENTRATION vs. FORECAST ERROR, SPEARMAN CORRELATION\n")
    f.write("=" * 70 + "\n\n")
    f.write(
        "Menguji apakah model yang mengonsentrasikan atribusi lebih tajam\n"
        "(pada komponen Trend atau timestep paling baru) mencapai error\n"
        "lebih rendah. Rho negatif berarti konsentrasi lebih tinggi\n"
        "berkaitan dengan MAE lebih rendah.\n\n"
    )
    for rec in records:
      f.write(f"{rec['label']}\n")
      f.write(f"  n            : {rec['n']}\n")
      f.write(f"  Spearman rho : {rec['rho']:.4f}\n")
      f.write(f"  p-value      : {rec['pval']:.4f}\n\n")
  print(f"  [+] Saved: {filename}")


# ============================================================
# SECTION 7: GRAPHICAL REPORTS
# ============================================================

def plot_feature_attribution(name, shap_vals, ig_vals, save_dir):
  os.makedirs(save_dir, exist_ok=True)
  x = np.arange(len(FEATURE_NAMES))
  width = 0.35
  fig, ax = plt.subplots(figsize=(9, 5))
  ax.bar(x - width / 2, shap_vals, width,
         label='SHAP', color=MODEL_COLORS[name])
  ax.bar(x + width / 2, ig_vals,   width,
         label='Integrated Gradients', color='gray', alpha=0.7)
  ax.set_xticks(x)
  ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha='right')
  ax.set_ylabel('Mean |Attribution|')
  ax.set_title(f'{name}, Feature Attribution')
  ax.legend()
  fig.tight_layout()
  fig.savefig(f"{save_dir}/{name.lower()}_feature_attribution.png", dpi=DPI)
  plt.close(fig)


def plot_timestep_attribution(name, shap_vals, ig_vals, save_dir):
  os.makedirs(save_dir, exist_ok=True)
  lags = [f"t-{LOOKBACK - 1 - t}" for t in range(LOOKBACK)]
  x = np.arange(LOOKBACK)
  fig, ax = plt.subplots(figsize=(9, 5))
  ax.plot(x, shap_vals, marker='o', label='SHAP', color=MODEL_COLORS[name])
  ax.plot(x, ig_vals,   marker='s',
          label='Integrated Gradients', color='gray', alpha=0.7)
  ax.set_xticks(x)
  ax.set_xticklabels(lags)
  ax.set_xlabel('Lookback timestep (t-9 = oldest, t-0 = terbaru)')
  ax.set_ylabel('Mean |Attribution| (dijumlah antar fitur)')
  ax.set_title(f'{name}, Timestep Attribution')
  ax.legend()
  fig.tight_layout()
  fig.savefig(f"{save_dir}/{name.lower()}_timestep_attribution.png", dpi=DPI)
  plt.close(fig)


def plot_attribution_heatmap(name, attr_arr, save_dir):
  """attr_arr: (N, T, 4) raw SHAP attributions for one model -> mean |attr|
  heatmap of shape (T, 4)."""
  os.makedirs(save_dir, exist_ok=True)
  heat = np.abs(attr_arr).mean(axis=0)
  fig, ax = plt.subplots(figsize=(7, 6))
  im = ax.imshow(heat, aspect='auto', cmap='viridis')
  ax.set_xticks(range(len(FEATURE_NAMES)))
  ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha='right')
  ax.set_yticks(range(LOOKBACK))
  ax.set_yticklabels([f"t-{LOOKBACK - 1 - t}" for t in range(LOOKBACK)])
  ax.set_title(f'{name}, SHAP Attribution Heatmap (timestep x fitur)')
  fig.colorbar(im, ax=ax, label='Mean |SHAP value|')

  vmax = heat.max()
  for t in range(heat.shape[0]):
    for f in range(heat.shape[1]):
      val = heat[t, f]
      color = 'white' if val < vmax * 0.55 else 'black'
      ax.text(f, t, f'{val:.3f}', ha='center', va='center',
              fontsize=7, color=color)

  fig.tight_layout()
  fig.savefig(f"{save_dir}/{name.lower()}_heatmap.png", dpi=DPI)
  plt.close(fig)


def plot_feature_importance_comparison(feature_shap, filename):
  x = np.arange(len(FEATURE_NAMES))
  width = 0.1
  fig, ax = plt.subplots(figsize=(11, 6))
  for i, name in enumerate(MODEL_NAMES):
    norm = feature_shap[name] / (feature_shap[name].sum() + 1e-12)
    ax.bar(x + (i - 3.5) * width, norm, width,
           label=name, color=MODEL_COLORS[name])
  ax.set_xticks(x)
  ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha='right')
  ax.set_ylabel('Share of total |SHAP| attribution')
  ax.set_title('Feature Attribution Share, All 8 Models')
  ax.legend(ncol=4, fontsize=8)
  fig.tight_layout()
  fig.savefig(filename, dpi=DPI)
  plt.close(fig)
  print(f"  [+] Saved: {filename}")


def plot_concentration_vs_error(records, split_label, filename):
  fig, ax = plt.subplots(figsize=(8, 6))
  for rec in records:
    ax.scatter(rec['concentration'], rec['mae'], s=80,
               color=MODEL_COLORS[rec['model']], label=rec['model'])
    ax.annotate(rec['model'], (rec['concentration'], rec['mae']),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
  ax.set_xlabel('Feature-attribution concentration (1 = fully on one feature)')
  ax.set_ylabel(f'{split_label} MAE')
  ax.set_title(f'Attribution Concentration vs. Forecast Error, {split_label}')
  fig.tight_layout()
  fig.savefig(filename, dpi=DPI)
  plt.close(fig)
  print(f"  [+] Saved: {filename}")


# ============================================================
# SECTION 8: PIPELINE
# ============================================================

def subsample(X_arr, n, seed=SEED):
  if len(X_arr) <= n:
    return X_arr
  rng = np.random.default_rng(seed)
  idx = rng.choice(len(X_arr), size=n, replace=False)
  return X_arr[idx]


def run_split(split_name, X_seq, X_train_seq, y_true_inv, scaler_y, device,
              feature_shap, feature_ig, timestep_shap, timestep_ig,
              concentration_records, sample_heatmap_arrs):
  print(f"\n{'='*60}")
  print(f"[*] Explaining {split_name} set "
        f"({len(X_seq)} windows, capped sample size {XAI_EVAL_SAMPLE_SIZE})")
  print(f"{'='*60}")

  X_seq_s = subsample(X_seq, XAI_EVAL_SAMPLE_SIZE)
  X_flat_s = X_seq_s[:, -1, :]
  y_eval_inv = subsample(y_true_inv.reshape(-1, 1), XAI_EVAL_SAMPLE_SIZE).flatten()

  X_train_flat = X_train_seq[:, -1, :]

  for name in MODEL_NAMES:
    print(f"\n[*] {name}...")
    model = load_trained_model(name, device)
    is_seq = name in SEQUENCE_MODELS
    X_eval = X_seq_s if is_seq else X_flat_s
    X_bg = subsample(X_train_seq if is_seq else X_train_flat, SHAP_BACKGROUND_SIZE)
    repeats = STABILITY_REPEATS[name]

    sv = compute_shap_attributions(model, X_bg, X_eval, device, repeats)
    iv = compute_ig_attributions(model, X_eval, device, repeats)

    f_shap = aggregate_feature_importance(sv)
    f_ig = aggregate_feature_importance(iv)
    feature_shap.setdefault(split_name, {})[name] = f_shap
    feature_ig.setdefault(split_name, {})[name] = f_ig

    t_shap = aggregate_timestep_importance(sv)
    t_ig = aggregate_timestep_importance(iv)
    if t_shap is not None:
      timestep_shap.setdefault(split_name, {})[name] = t_shap
      timestep_ig.setdefault(split_name, {})[name] = t_ig
      sample_heatmap_arrs.setdefault(split_name, {})[name] = sv

    preds_inv = scaler_y.inverse_transform(
        predict_array(model, X_eval, device).reshape(-1, 1)
    ).flatten()
    metrics = compute_metrics(y_eval_inv, preds_inv)

    f_conc = concentration_score(f_shap)
    t_conc = concentration_score(t_shap) if t_shap is not None else None
    concentration_records.append({
        'split': split_name, 'model': name,
        'feature_concentration_shap': f_conc,
        'timestep_concentration_shap': t_conc,
        'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'],
    })
    print(f"    feature_concentration={f_conc:.3f}  "
          f"timestep_concentration={t_conc if t_conc is None else round(t_conc, 3)}  "
          f"MAE={metrics['MAE']:.3f}")


def main():
  os.makedirs(STAT_DIR, exist_ok=True)
  log_path = f"{STAT_DIR}/00_xai_log.txt"
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
    print(f"[+] Full XAI log saved: {log_path}")


def _run_pipeline():
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # cuDNN's fused RNN/LSTM/GRU backward kernel refuses to run unless the
  # forward pass was also in training mode (which would re-enable dropout
  # and make attributions stochastic). Disabling cuDNN for RNN ops falls
  # back to the generic implementation, which supports backward in eval
  # mode and keeps attributions deterministic; the cost is negligible at
  # this model/data scale.
  torch.backends.cudnn.enabled = False

  print("=" * 66)
  print(f"  STAGE 09: XAI EXPLAINABILITY, run={TAG}")
  print(f"  Device: {device} | Seed: {SEED} | Lookback T: {LOOKBACK}")
  print("=" * 66)

  for d in [STAT_DIR, GRAPH_DIR, FEATURE_ATTR_DIR, TIMESTEP_ATTR_DIR,
            HEATMAP_DIR, COMPARISON_DIR]:
    os.makedirs(d, exist_ok=True)

  print("\n[*] Loading data...")
  splits, _, scaler_y, _ = load_data()
  split_keys = [k for k in ["test", "unseen"] if k in splits]

  X_train_seq, _ = splits["train"]

  def inv_y(arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

  y_true_inv = {sk: inv_y(splits[sk][1]) for sk in split_keys}

  feature_shap, feature_ig = {}, {}
  timestep_shap, timestep_ig = {}, {}
  concentration_records = []
  sample_heatmap_arrs = {}

  for sk in split_keys:
    X_seq, _ = splits[sk]
    run_split(sk, X_seq, X_train_seq, y_true_inv[sk], scaler_y, device,
              feature_shap, feature_ig, timestep_shap, timestep_ig,
              concentration_records, sample_heatmap_arrs)

  # --------------------------------------------------------
  # Statistical reports
  # --------------------------------------------------------
  print("\n[*] Saving statistical reports...")
  for sk in split_keys:
    save_feature_attribution_report(
        feature_shap[sk], feature_ig[sk],
        f"{STAT_DIR}/01_feature_attribution_{sk}.csv")
    save_timestep_attribution_report(
        timestep_shap[sk], timestep_ig[sk],
        f"{STAT_DIR}/02_timestep_attribution_{sk}.csv")

  save_concentration_report(concentration_records,
                            f"{STAT_DIR}/03_concentration_metrics.csv")

  corr_records = []
  for sk in split_keys:
    split_recs = [r for r in concentration_records if r['split'] == sk]
    f_conc = [r['feature_concentration_shap'] for r in split_recs]
    maes = [r['MAE'] for r in split_recs]
    rho, pval = spearmanr(f_conc, maes)
    corr_records.append({
        'label': f"[{sk}] Feature concentration vs MAE (8 model)",
        'n': len(split_recs), 'rho': rho, 'pval': pval,
    })

    seq_recs = [r for r in split_recs if r['model'] in SEQUENCE_MODELS]
    t_conc = [r['timestep_concentration_shap'] for r in seq_recs]
    t_maes = [r['MAE'] for r in seq_recs]
    rho2, pval2 = spearmanr(t_conc, t_maes)
    corr_records.append({
        'label': f"[{sk}] Timestep concentration vs MAE (7 model sequence)",
        'n': len(seq_recs), 'rho': rho2, 'pval': pval2,
    })
  save_correlation_report(
      corr_records, f"{STAT_DIR}/04_concentration_vs_error_correlation.txt")

  # --------------------------------------------------------
  # Graphical reports
  # --------------------------------------------------------
  print("\n[*] Saving graphical reports...")
  for sk in split_keys:
    for name in MODEL_NAMES:
      plot_feature_attribution(
          name, feature_shap[sk][name], feature_ig[sk][name],
          f"{FEATURE_ATTR_DIR}/{sk}")
    for name in SEQUENCE_MODELS:
      plot_timestep_attribution(
          name, timestep_shap[sk][name], timestep_ig[sk][name],
          f"{TIMESTEP_ATTR_DIR}/{sk}")
      plot_attribution_heatmap(
          name, sample_heatmap_arrs[sk][name], f"{HEATMAP_DIR}/{sk}")

    plot_feature_importance_comparison(
        feature_shap[sk], f"{COMPARISON_DIR}/feature_importance_all_models_{sk}.png")

    split_recs = [r for r in concentration_records if r['split'] == sk]
    scatter_recs = [{'model': r['model'], 'concentration': r['feature_concentration_shap'],
                     'mae': r['MAE']} for r in split_recs]
    plot_concentration_vs_error(
        scatter_recs, sk, f"{COMPARISON_DIR}/concentration_vs_mae_{sk}.png")

  print(f"\n[+] Done. Run tag: {TAG}. Outputs in {STAT_DIR} and {GRAPH_DIR}.")


if __name__ == "__main__":
  main()
