# Final Task DL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replikasi training 8 model DL plus XAI dari `experiment-argument-ID.md` pada seluruh 9.681 baris dataset CEEMDAN lokal, lalu benchmarking kontekstual terhadap Wu et al. (ICEEMDAN-SCA-RVFL) pada rentang data yang sama dengan paper tersebut, dengan laporan akhir berformat .md.

**Architecture:** Dua run paralel dengan pipeline identik. Run `full` memakai semua baris `dataset/WTI-CEEMDAN-FE-n10.csv` dengan split kronologis 80/10/10 (train/test/unseen), run `wu` memakai subset `Date <= 2020-02-10` dengan split 80/20 (train/test) mengikuti setup Wu et al. Delapan arsitektur (MLP, RNN, LSTM, BiLSTM, GRU, TCN, Transformer, Informer) diambil verbatim dari repo referensi, hanya kontrak inputnya diadaptasi dari 7 fitur makro menjadi 4 komponen CEEMDAN (Trend, group_1, group_2, res) dengan lookback T=10. XAI (SHAP GradientExplainer + Integrated Gradients, atribusi fitur/timestep, concentration score, korelasi Spearman vs MAE) direplikasi persis dari `09-xai-explainability.py`.

**Tech Stack:** venv di `D:\Coding\#bigdata\venv` (Python 3.11.9, torch 2.5.1+cu121 dengan CUDA aktif, shap 0.51.0, captum 0.9.0, sklearn 1.8.0, pandas 2.3.3, matplotlib 3.10.8). JANGAN pakai python global (torch 2.9.0 CPU only, tanpa shap/captum).

**Mode eksekusi:** semua perintah `python ...` di plan ini DIJALANKAN SENDIRI OLEH USER di cmd miliknya. Agent menyiapkan script, lalu berhenti di step "Run" dan menunggu user melaporkan hasil/output sebelum lanjut ke step verifikasi. Aktivasi venv di cmd:

```bat
D:\Coding\#bigdata\venv\Scripts\activate.bat
```

## Global Constraints

Dari `task.txt`, berlaku untuk semua task.

1. Penulisan laporan .md TIDAK boleh memakai tanda "-", "--", "---" dan sejenisnya sebagai tanda baca naratif. Ganti dengan titik atau koma. Kalimat dengan ":" dijadikan baris baru berbentuk list. (Bullet list markdown `-` sebagai struktur list tetap boleh, larangan ini untuk dash sebagai pemisah kalimat.)
2. Semua summary dan report berformat `.md`.
3. Struktur folder disamakan dengan repo referensi `D:\Coding\#bigdata\crude-oil-forecasting-DL` (dataset/splits, dataset/scalers, models, evaluations/statistical/model-train, evaluations/graphical/model-train, evaluations/xai/statistical, evaluations/xai/graphical). Karena ada dua run, tambahkan satu level subfolder `{full,wu}` di dalam tiap direktori output.
4. Script diambil dan diadaptasi langsung dari repo referensi. Sumber utama:
   - `D:\Coding\#bigdata\crude-oil-forecasting-DL\05-dl-model-training.py` (7 model + training loop + metrik + DM test + plot)
   - `D:\Coding\#bigdata\crude-oil-forecasting-DL\08-rnn-model-training.py` (kelas RNNModel)
   - `D:\Coding\#bigdata\crude-oil-forecasting-DL\09-xai-explainability.py` (XAI lengkap)
   - `D:\Coding\#bigdata\crude-oil-forecasting-DL\04-fe-and-split.py` (pola split/winsorize/scaling)
5. Benchmark pembanding adalah `references/Wu-Improved-CEEMDAN-SCA-RVFL.md`. Angka acuan horizon 1 dari Table 4: MAPE 0.0035, RMSE 0.2801, Dstat 0.9273.
6. Seed 42, LOOKBACK 10, konfigurasi training sama dengan referensi (BATCH_SIZE 32, MAX_EPOCHS 200, PATIENCE 20, LR 1e-3, ReduceLROnPlateau factor 0.5 patience 10 min 1e-5, VAL_FRACTION 0.10).

## Fakta Dataset (sudah diverifikasi)

- File: `dataset/WTI-CEEMDAN-FE-n10.csv`, 9.681 baris data, 1988-01-11 sampai 2026-06-29.
- Kolom per komponen `{Trend, group_1, group_2, res}`: `<comp>_lag_10` ... `<comp>_lag_1` (10 lag) plus `target_<comp>`, ditambah `Date` dan `actual_close`. Total 46 kolom.
- Sudah pre-windowed. Satu baris = satu window lengkap. TIDAK perlu sliding window seperti `create_sequences` di script referensi.
- Jumlah `target_Trend + target_group_1 + target_group_2 + target_res` = `actual_close` (properti aditif CEEMDAN). Wajib di-assert di script prep.
- Urutan timestep: `lag_10` adalah t-9 (terlama), `lag_1` adalah t-0 (terbaru). Reshape harus menjaga urutan ini.
- Keputusan desain: replikasi "persis seperti experiment-argument-ID.md" berarti SATU model per arsitektur dengan input multivariat `(batch, 10, 4)` dan target `actual_close` langsung. Bukan skema per-komponen 8x4 model seperti repo thesis. XAI kemudian menjawab pertanyaan yang sama, fitur mana (komponen CEEMDAN mana) dan timestep mana yang dominan.

## Konteks Pembanding Wu (untuk laporan)

Wajib disebut jujur di laporan benchmark, bukan disembunyikan.

1. Wu memakai 8.596 sampel (2 Jan 1986 sampai 10 Feb 2020) dengan split 80/20. Subset kita `Date <= 2020-02-10` berisi lebih sedikit baris (dataset mulai 1988-01-11 karena warmup 500 baris expanding window di pipeline thesis tidak menghasilkan dekomposisi). Jumlah pasti dicetak oleh script prep.
2. Dekomposisi kita expanding window tanpa look-ahead. Dekomposisi Wu dilakukan sekali pada seluruh series (termasuk data test) sehingga mengandung look-ahead leakage. Ini membuat angka Wu secara struktural lebih optimis.
3. Fitur input berbeda (kami 4 komponen CEEMDAN hasil LOWESS+CEEMDAN expanding, Wu memakai IMF ICEEMDAN penuh dengan RVFL per IMF).
4. Mapping metrik: MAPE Wu adalah fraksi (MAPE% kita dibagi 100), Dstat Wu adalah fraksi dari DA% kita.

---

### Task 0: Environment, dependency, scaffold, git init

**Files:**
- Create: `.gitignore`
- Create: folder skeleton (lihat step 3)

**Interfaces:**
- Produces: environment dengan `shap` dan `captum` terinstall, repo git terinisialisasi, folder output siap dipakai task berikutnya.

- [ ] **Step 1: Verifikasi venv (tidak perlu install apa pun)**

Sudah diverifikasi 2026-07-13: `D:\Coding\#bigdata\venv` berisi torch 2.5.1+cu121 (CUDA available), shap 0.51.0, captum 0.9.0. Cukup pastikan user menjalankan semua script dengan venv ini aktif:
```bat
D:\Coding\#bigdata\venv\Scripts\activate.bat
python -c "import torch, shap, captum; print(torch.cuda.is_available())"
```
Expected: `True`.

- [ ] **Step 2: Buat .gitignore**

```gitignore
__pycache__/
*.pt
*.pkl
dataset/splits/
dataset/scalers/
```

- [ ] **Step 3: Buat folder skeleton**

Run:
```powershell
$dirs = @(
  'dataset/splits/full','dataset/splits/wu',
  'dataset/scalers/full','dataset/scalers/wu',
  'models/full','models/wu',
  'evaluations/statistical/model-train/full','evaluations/statistical/model-train/wu',
  'evaluations/graphical/model-train/full','evaluations/graphical/model-train/wu',
  'evaluations/xai/statistical/full','evaluations/xai/statistical/wu',
  'evaluations/xai/graphical/full','evaluations/xai/graphical/wu'
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force $_ } | Out-Null
Get-ChildItem -Recurse -Directory | Select-Object FullName
```
Expected: semua folder tercetak.

- [ ] **Step 4: Git init dan commit awal**

```powershell
git init
git add .
git commit -m "chore: scaffold final task DL project"
```

---

### Task 1: Script prep dan split `04-fe-and-split.py`

**Files:**
- Create: `04-fe-and-split.py`

**Interfaces:**
- Consumes: `dataset/WTI-CEEMDAN-FE-n10.csv`
- Produces per run tag `{full,wu}`:
  - `dataset/splits/<tag>/splits.npz` berisi key `X_train, y_train, X_test, y_test, dates_train, dates_test` dan untuk `full` juga `X_unseen, y_unseen, dates_unseen`. X shape `(N, 10, 4)` float32 sudah winsorized+scaled, y shape `(N,)` float32 scaled.
  - `dataset/scalers/<tag>/scaler_X.pkl` (MinMaxScaler fit pada `(N*10, 4)` train) dan `scaler_y.pkl` (MinMaxScaler fit pada y train).
  - `dataset/splits/<tag>/split-report.md`

- [ ] **Step 1: Tulis script lengkap**

Isi `04-fe-and-split.py`:

```python
# ==================================================
# STAGE 04: FEATURE PREP & CHRONOLOGICAL SPLIT
# Final Task DL. Input pre-windowed CEEMDAN feature dataset.
# Run tags:
#   full : semua baris, split 80/10/10 (train/test/unseen)
#   wu   : Date <= 2020-02-10, split 80/20 (train/test), setup Wu et al.
# ==================================================
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEED = 42
LOOKBACK = 10
COMPONENTS = ["Trend", "group_1", "group_2", "res"]
N_FEATURES = len(COMPONENTS)
WU_CUTOFF = "2020-02-10"
DATA_PATH = "dataset/WTI-CEEMDAN-FE-n10.csv"


def build_xy(df):
    """Reshape pre-windowed rows to (N, T=10, F=4). lag_10 = t-9 (oldest),
    lag_1 = t-0 (newest)."""
    N = len(df)
    X = np.zeros((N, LOOKBACK, N_FEATURES), dtype=np.float64)
    for f, comp in enumerate(COMPONENTS):
        for t in range(LOOKBACK):
            lag = LOOKBACK - t
            X[:, t, f] = df[f"{comp}_lag_{lag}"].values
    y = df["actual_close"].values.astype(np.float64)
    return X, y


def winsorize(X_train, others, lo_q=0.01, hi_q=0.99):
    """Clip per feature using train quantiles computed over all timesteps."""
    lo = np.quantile(X_train.reshape(-1, N_FEATURES), lo_q, axis=0)
    hi = np.quantile(X_train.reshape(-1, N_FEATURES), hi_q, axis=0)
    clipped = [np.clip(X_train, lo, hi)]
    clipped += [np.clip(Xo, lo, hi) for Xo in others]
    return clipped, lo, hi


def scale(X_train, others, y_train, y_others):
    scaler_X = MinMaxScaler(clip=True)
    scaler_X.fit(X_train.reshape(-1, N_FEATURES))
    Xs = [scaler_X.transform(X.reshape(-1, N_FEATURES)).reshape(X.shape).astype(np.float32)
          for X in [X_train] + others]
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))
    ys = [scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
          for y in [y_train] + y_others]
    return Xs, ys, scaler_X, scaler_y


def main(tag):
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # integrity: additive CEEMDAN reconstruction must equal actual_close
    recon = sum(df[f"target_{c}"] for c in COMPONENTS)
    max_err = (recon - df["actual_close"]).abs().max()
    assert max_err < 1e-6, f"additive check failed, max err {max_err}"
    print(f"[OK] additive reconstruction check, max abs err = {max_err:.2e}")

    if tag == "wu":
        df = df[df["Date"] <= WU_CUTOFF].reset_index(drop=True)
        print(f"[wu] rows <= {WU_CUTOFF}: {len(df)} "
              f"(Wu et al. memakai 8596 sampel 1986-01-02 s.d. 2020-02-10, "
              f"selisih karena warmup expanding window, wajib disebut di laporan)")

    X, y = build_xy(df)
    dates = df["Date"].dt.strftime("%Y-%m-%d").values

    n = len(df)
    if tag == "full":
        n_train = int(n * 0.8)
        n_test = int((n - n_train) * 0.5)
        idx = {"train": slice(0, n_train),
               "test": slice(n_train, n_train + n_test),
               "unseen": slice(n_train + n_test, n)}
    else:
        n_train = int(n * 0.8)
        idx = {"train": slice(0, n_train), "test": slice(n_train, n)}

    parts = {k: (X[s], y[s], dates[s]) for k, s in idx.items()}
    train_X, train_y, _ = parts["train"]
    other_keys = [k for k in parts if k != "train"]

    clipped, lo, hi = winsorize(train_X, [parts[k][0] for k in other_keys])
    Xs, ys, scaler_X, scaler_y = scale(
        clipped[0], clipped[1:], train_y, [parts[k][1] for k in other_keys])

    out = {}
    for i, k in enumerate(["train"] + other_keys):
        out[f"X_{k}"] = Xs[i]
        out[f"y_{k}"] = ys[i]
        out[f"dates_{k}"] = parts[k][2]

    split_dir = f"dataset/splits/{tag}"
    scaler_dir = f"dataset/scalers/{tag}"
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    np.savez(f"{split_dir}/splits.npz", **out)
    joblib.dump(scaler_X, f"{scaler_dir}/scaler_X.pkl")
    joblib.dump(scaler_y, f"{scaler_dir}/scaler_y.pkl")

    lines = [f"# Split Report ({tag})", "",
             f"Total baris: {n}", "",
             "Rentang per split:", ""]
    for k in ["train"] + other_keys:
        d = parts[k][2]
        lines.append(f"- {k}: {d[0]} s.d. {d[-1]}, {len(d)} baris")
    lines += ["", "Winsorize kuantil train per fitur:", ""]
    for f, c in enumerate(COMPONENTS):
        lines.append(f"- {c}: clip [{lo[f]:.4f}, {hi[f]:.4f}]")
    lines += ["", "Scaling:", "", "- MinMaxScaler fit pada train saja, clip=True untuk X"]
    with open(f"{split_dir}/split-report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    for k in ["train"] + other_keys:
        print(f"[{tag}] {k}: {out[f'X_{k}'].shape}, {parts[k][2][0]} .. {parts[k][2][-1]}")
    print(f"[OK] saved {split_dir}/splits.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["full", "wu"], required=True)
    args = ap.parse_args()
    np.random.seed(SEED)
    main(args.run)
```

- [ ] **Step 2: Jalankan untuk kedua tag**

Run:
```powershell
python 04-fe-and-split.py --run full
python 04-fe-and-split.py --run wu
```
Expected untuk `full`: additive check OK, train ~7.744 baris, test ~968, unseen ~969, file npz + 2 pkl + split-report.md tertulis. Expected untuk `wu`: jumlah baris subset tercetak (sekitar 8.000, catat angka pastinya untuk laporan), train 80% test 20%.

- [ ] **Step 3: Sanity check manual**

Run:
```powershell
python -c "import numpy as np; d=np.load('dataset/splits/full/splits.npz'); print({k: d[k].shape for k in d.files}); print('X range', d['X_train'].min(), d['X_train'].max())"
```
Expected: X_train shape `(N, 10, 4)`, nilai dalam [0, 1].

- [ ] **Step 4: Commit**

```powershell
git add 04-fe-and-split.py dataset/splits/full/split-report.md dataset/splits/wu/split-report.md
git commit -m "feat: data prep and chronological split for full and wu runs"
```

---

### Task 2: Script training `05-dl-model-training.py` (8 arsitektur)

**Files:**
- Create: `05-dl-model-training.py` (adaptasi dari `D:\Coding\#bigdata\crude-oil-forecasting-DL\05-dl-model-training.py`)

**Interfaces:**
- Consumes: `dataset/splits/<tag>/splits.npz`, `dataset/scalers/<tag>/scaler_y.pkl`
- Produces per tag:
  - `models/<tag>/{mlp,rnn,lstm,bilstm,gru,tcn,transformer,informer}_model.pt` (state_dict)
  - `evaluations/statistical/model-train/<tag>/01_metrics_summary.md` (tabel MAE, MAPE%, SMAPE%, RMSE, DA%, MASE, R2 per split)
  - `evaluations/statistical/model-train/<tag>/02_dm_test.md` (matriks DM test antar model)
  - `evaluations/statistical/model-train/<tag>/predictions_<split>.csv` (Date, actual, kolom prediksi per model, skala asli USD)
  - `evaluations/graphical/model-train/<tag>/actual-vs-predicted-*/` plot per model, `metrics_comparison.png`
  - `evaluations/statistical/model-train/<tag>/run_log.txt`

- [ ] **Step 1: Salin kerangka dari repo referensi**

Copy file sumber ke proyek ini lalu edit. Bagian yang WAJIB disalin verbatim (jangan ditulis ulang dari ingatan):
- Kelas model dari file sumber `05-dl-model-training.py`: `MLPModel` (baris 166-178), `LSTMModel` (181-191), `BiLSTMModel` (194-204), `GRUModel` (207-217), `CausalConv1d`+`TCNBlock`+`TCNModel` (222-276), `PositionalEncoding`+`TransformerModel` (281-319), `ProbSparseAttention`+`InformerEncoderLayer`+`InformerModel` (324-415).
- Kelas `RNNModel` dari `08-rnn-model-training.py` baris 161-172.
- Seluruh training loop (early stopping + ReduceLROnPlateau), fungsi metrik (MAE, MAPE, SMAPE, RMSE, DA, MASE, R2), fungsi DM test (Harvey et al. 1997), dan fungsi plotting dari file sumber.
- Kelas `Tee` untuk logging.

- [ ] **Step 2: Modifikasi konfigurasi**

Ganti SECTION 1 di salinan dengan:

```python
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--run", choices=["full", "wu"], required=True)
ap.add_argument("--max-epochs", type=int, default=200)  # smoke test bisa pakai 2
ARGS = ap.parse_args()
TAG = ARGS.run

SEED = 42
LOOKBACK = 10
N_FEATURES = 4
BATCH_SIZE = 32
MAX_EPOCHS = ARGS.max_epochs
PATIENCE = 20
LR = 1e-3
LR_FACTOR = 0.5
LR_PATIENCE = 10
LR_MIN = 1e-5
VAL_FRACTION = 0.10
DPI = 300

DATASET_DIR = f"dataset/splits/{TAG}"
SCALER_DIR = f"dataset/scalers/{TAG}"
STAT_DIR = f"evaluations/statistical/model-train/{TAG}"
GRAPH_DIR = f"evaluations/graphical/model-train/{TAG}"
MODEL_DIR = f"models/{TAG}"

MODEL_NAMES = ["MLP", "RNN", "LSTM", "BiLSTM",
               "GRU", "TCN", "Transformer", "Informer"]
FEATURE_NAMES = ["Trend", "IMF_Group1", "IMF_Group2", "Residual"]
```

- [ ] **Step 3: Ganti SECTION 2 (loading), hapus pembuatan window**

Dataset sudah pre-windowed, jadi `create_sequences` dan `create_eval_sequences` dari sumber DIHAPUS dan diganti:

```python
def load_data():
    d = np.load(f"{DATASET_DIR}/splits.npz", allow_pickle=True)
    scaler_y = joblib.load(f"{SCALER_DIR}/scaler_y.pkl")
    has_unseen = "X_unseen" in d.files
    splits = {
        "train": (d["X_train"], d["y_train"], pd.to_datetime(d["dates_train"])),
        "test": (d["X_test"], d["y_test"], pd.to_datetime(d["dates_test"])),
    }
    if has_unseen:
        splits["unseen"] = (d["X_unseen"], d["y_unseen"], pd.to_datetime(d["dates_unseen"]))
    return splits, scaler_y, has_unseen
```

Semua evaluasi unseen di script dibungkus `if has_unseen:` (run `wu` tidak punya unseen). MLP tetap menerima input `(batch, 4)` dengan mengambil timestep terakhir, ikuti pola file sumber (di sana MLP diberi `X[:, -1, :]`, pertahankan mekanisme yang sama, cek bagian train/eval loop file sumber saat menyalin).

- [ ] **Step 4: Output metrik ke .md, prediksi inverse-transform**

Semua metrik dihitung pada skala asli USD setelah `scaler_y.inverse_transform`, sama seperti sumber. Ganti penulisan `01_metrics_summary.txt` menjadi `01_metrics_summary.md` berbentuk tabel markdown per split (kolom: Model, MAE, MAPE%, SMAPE%, RMSE, DA%, MASE, R2), dan `02_dm_test.md` untuk matriks DM. Tambahkan penulisan `predictions_<split>.csv` berisi Date, actual, prediksi per model.

- [ ] **Step 5: Smoke test 2 epoch**

Run:
```powershell
python 05-dl-model-training.py --run full --max-epochs 2
```
Expected: 8 model terlatih tanpa exception, file `01_metrics_summary.md`, `02_dm_test.md`, plot, dan 8 file `.pt` muncul di folder `full`. Nilai metrik boleh jelek, yang dicek hanya pipeline jalan.

- [ ] **Step 6: Hapus artefak smoke test, jalankan run full sesungguhnya**

Run:
```powershell
Remove-Item models/full/*.pt
python 05-dl-model-training.py --run full
```
Expected: selesai jauh lebih cepat di GPU venv (train ~7.700 window, referensi 4.208 window). Cek `01_metrics_summary.md` berisi 8 baris per split dan R2 test mayoritas model > 0.9 (pola referensi, LSTM/GRU terbaik).

- [ ] **Step 7: Jalankan run wu**

Run:
```powershell
python 05-dl-model-training.py --run wu
```
Expected: sama, tanpa bagian unseen.

- [ ] **Step 8: Commit**

```powershell
git add 05-dl-model-training.py evaluations/
git commit -m "feat: 8-architecture DL training for full and wu runs"
```

---

### Task 3: Script XAI `09-xai-explainability.py`

**Files:**
- Create: `09-xai-explainability.py` (adaptasi dari `D:\Coding\#bigdata\crude-oil-forecasting-DL\09-xai-explainability.py`)

**Interfaces:**
- Consumes: `models/<tag>/*.pt`, `dataset/splits/<tag>/splits.npz`, definisi kelas model identik dengan Task 2 (state_dict harus load tanpa key mismatch).
- Produces per tag di `evaluations/xai/statistical/<tag>/`:
  - `01_feature_attribution_<split>.csv`, `02_timestep_attribution_<split>.csv`, `03_concentration_metrics.csv`, `04_concentration_vs_error_correlation.md`
- Dan di `evaluations/xai/graphical/<tag>/`: subfolder `feature-attribution/`, `timestep-attribution/`, `attribution-heatmaps/`, `comparison/` berisi plot per model per split, sama seperti struktur referensi.

- [ ] **Step 1: Salin dan adaptasi**

Salin file sumber `09-xai-explainability.py` utuh, lalu ubah:
- Tambah `argparse --run {full,wu}` seperti Task 2, semua path diberi suffix `/{TAG}`.
- `N_FEATURES = 4`, `FEATURE_NAMES = ["Trend", "IMF_Group1", "IMF_Group2", "Residual"]`.
- Definisi kelas model diimpor dari script 05 (`from importlib import import_module` tidak perlu, cukup duplikasi kelas verbatim seperti file sumber melakukannya, atau `from dl_models import ...` bila kelas dipisah; ikuti pola file sumber yang menduplikasi kelas).
- Loading data memakai `load_data()` yang sama dengan Task 2 (background SHAP diambil dari akhir train, eval set dari test dan unseen, cap `XAI_EVAL_SAMPLE_SIZE = 200`, `SHAP_BACKGROUND_SIZE = 200`, `SHAP_NSAMPLES = 50`, `IG_STEPS = 50`, `STABILITY_REPEATS` Informer = 10, lainnya 1, semua sama dengan sumber).
- Untuk run `wu` yang tidak punya unseen, loop split hanya `["test"]`.
- Output `04_concentration_vs_error_correlation` ditulis sebagai `.md` (tabel markdown rho dan p-value Spearman).
- MLP dianalisis pada input `(batch, 4)` seperti di sumber (atribusi per fitur saja, tanpa timestep), model sequence pada `(batch, 10, 4)`.

- [ ] **Step 2: Jalankan untuk kedua run**

Run:
```powershell
python 09-xai-explainability.py --run full
python 09-xai-explainability.py --run wu
```
Expected: per model tercetak progres SHAP dan IG, semua CSV dan PNG muncul. Runtime di GPU venv diperkirakan beberapa menit per run (200 window × nsamples 50, Informer 10 repeat).

- [ ] **Step 3: Sanity check hasil**

Cek `01_feature_attribution_test.csv` run full berisi 8 baris × 4 kolom fitur, nilai non-negatif, dan `03_concentration_metrics.csv` berisi kolom feature_concentration dan timestep_concentration. Ekspektasi domain: atribusi `Trend` dominan (Trend adalah level harga, komponen lain osilasi kecil), catat apa pun hasilnya untuk laporan.

- [ ] **Step 4: Commit**

```powershell
git add 09-xai-explainability.py evaluations/xai/
git commit -m "feat: SHAP + Integrated Gradients XAI for both runs"
```

---

### Task 4: Laporan akhir .md

**Files:**
- Create: `final-task-report-ID.md` (laporan utama, bahasa Indonesia, bahan presentasi)
- Create: `benchmark-wu-comparison.md`
- Create: `README.md`

**Interfaces:**
- Consumes: semua output `evaluations/` kedua run, `references/Wu-Improved-CEEMDAN-SCA-RVFL.md`, angka Wu Table 4 dan Table 6.

- [ ] **Step 1: Tulis `final-task-report-ID.md`**

Struktur wajib (isi angka diambil dari file hasil, bukan dikarang):
1. Pendahuluan dan tujuan (replikasi experiment-argument-ID.md pada dataset CEEMDAN penuh 1988 sampai 2026, plus benchmark Wu).
2. Dataset dan preprocessing (fakta dataset di atas, split, winsorize, scaling, properti aditif CEEMDAN).
3. Delapan arsitektur, alasan pemilihan diringkas dari `experiment-argument-ID.md` dengan atribusi jelas.
4. Hasil training run full (tabel test dan unseen dari `01_metrics_summary.md`, DM test, plot utama disisipkan sebagai gambar relatif).
5. Hasil XAI run full (feature attribution, timestep attribution, concentration vs MAE, bandingkan polanya dengan temuan referensi, apakah LSTM tetap menang, apakah Transformer tetap diffus).
6. Kesimpulan.

Aturan penulisan wajib (Global Constraint 1):
- Tidak ada "-", "--", "---" sebagai tanda baca naratif, ganti titik atau koma.
- Kalimat berpola "X: a, b, c" ditulis ulang menjadi baris baru berbentuk list.

- [ ] **Step 2: Tulis `benchmark-wu-comparison.md`**

Struktur:
1. Setup Wu et al. (data 1986 sampai 2020, 8.596 sampel, 80/20, metrik MAPE, RMSE, Dstat).
2. Setup kami untuk run wu (jumlah baris aktual dari split-report, split 80/20, 8 model DL).
3. Tabel perbandingan horizon 1. Baris untuk ICEEMDAN-SCA-RVFL (MAPE 0.0035, RMSE 0.2801, Dstat 0.9273), EEMD-SCA-RVFL, dan model tunggal SCA-RVFL (MAPE 0.0157, RMSE 1.2183, Dstat 0.7522) dari Table 2 dan 4 paper, ditambah 8 baris model kami dengan MAPE fraksi (MAPE%/100), RMSE, dan Dstat (DA%/100) dari `evaluations/statistical/model-train/wu/01_metrics_summary.md`.
4. Bagian "Catatan kejujuran perbandingan" berisi 4 poin konteks pembanding Wu di atas (jumlah baris beda, look-ahead leakage di Wu, fitur beda, definisi metrik).
5. Interpretasi.

- [ ] **Step 3: Tulis `README.md`**

Ringkas: tujuan proyek, cara menjalankan pipeline (urutan `04 → 05 → 09` dengan `--run full` dan `--run wu`), struktur folder, daftar laporan. Ikuti aturan penulisan yang sama.

- [ ] **Step 4: Lint tanda baca**

Run:
```powershell
Select-String -Path final-task-report-ID.md,benchmark-wu-comparison.md,README.md -Pattern ' -- | — |---' | Select-Object Filename,LineNumber,Line
```
Expected: tidak ada match pada kalimat naratif (separator tabel markdown `|---|` dikecualikan, cek manual hasilnya).

- [ ] **Step 5: Commit**

```powershell
git add final-task-report-ID.md benchmark-wu-comparison.md README.md
git commit -m "docs: final report, Wu benchmark comparison, README"
```

---

## Self-Review Checklist (sudah dijalankan saat menulis plan)

1. Spec coverage: task.txt tugas 1 (baca laporan) selesai saat analisis. Tugas 2 (training + XAI semua row) = Task 1, 2, 3 run full. Tugas 3 (benchmark Wu, row disamakan) = Task 1, 2, 3 run wu + Task 4 step 2. Tugas 4 (script dari repo referensi) = sumber verbatim di Task 2 dan 3. Requirement 2 (tanda baca) = Task 4 step 4. Requirement 4 (.md) = semua output laporan .md. Requirement 5 (struktur folder) = Task 0 step 3.
2. Placeholder scan: tidak ada TBD. Kode yang tidak ditulis penuh di plan merujuk file sumber dengan path dan baris eksak, sesuai instruksi task.txt point 4 bahwa script diambil langsung dari repo referensi.
3. Konsistensi tipe: `load_data()` dipakai identik di Task 2 dan 3, key npz konsisten dengan Task 1, nama checkpoint `*_model.pt` konsisten Task 2 → Task 3.

## Risiko yang diketahui

1. Training memakai GPU via venv `D:\Coding\#bigdata\venv` (CUDA aktif). Estimasi run full 8 model di GPU jauh lebih cepat dari CPU, kemungkinan di bawah 1 jam. Semua run dieksekusi user sendiri di cmd.
2. Distribusi komponen `group_1` bergeser besar antar era (nilai 2026 mencapai −35 vs era 1988 sekitar ±0.5). MinMax fit di train mencakup rentang lebar, winsorize 1%/99% menahan ekstrem. Kalau prediksi test/unseen jenuh di batas clip, catat di laporan sebagai temuan, jangan diam-diam ganti scaler.
3. Angka Wu tidak bisa disamai secara apples-to-apples (leakage dekomposisi mereka). Laporan wajib memuat catatan kejujuran, bukan mengklaim kalah/menang mentah-mentah.
