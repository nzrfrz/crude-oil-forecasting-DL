# ==================================================
# Mohammad Nizar Farizi
# 25.52.1805
#
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
