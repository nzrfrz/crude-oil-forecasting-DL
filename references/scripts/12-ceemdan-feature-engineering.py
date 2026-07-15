"""
12-ceemdan-feature-engineering.py

Tahap 2 (CEEMDAN) - langkah 6a, lihat modeling-plan-ceemdan.md.

Windowing/lag PER KOMPONEN (Trend, group_1, group_2, res) dari dataset
expanding-window final (hasil script 10). Untuk tiap komponen dan tiap
panjang window n, dibentuk pasangan supervised:
  X = [komponen_{t-n}, ..., komponen_{t-1}]  (lag_1..lag_n)
  y = komponen_t                              (target_<komponen>)

Sengaja dijalankan SEBELUM split (langkah 6b) -- lag cuma menoleh ke
belakang jadi tidak ada risiko leakage di urutan manapun, tapi kalau
windowing dibuat SETELAH split (per partisi terpisah), baris-baris awal
Test/Unseen akan kehilangan akses ke lag dari ekor partisi sebelumnya.
Windowing di sini dilakukan di atas SATU series kontinu dulu supaya
tidak ada baris yang hilang di batas partisi (lihat modeling-plan-
ceemdan.md langkah 6a untuk penjelasan lengkap).

`actual_close` disertakan di tiap baris HANYA sebagai referensi ground-
truth untuk evaluasi akhir setelah rekonstruksi aditif (P_hat = jumlah
semua komponen hasil forecast) -- BUKAN target training untuk model
manapun (lihat author-QnA.txt entri 2026-07-06 soal peran actual_close).
Tidak ada konsep "anchor_close" seperti pipeline log-return karena
rekonstruksi di sini aditif langsung, tidak perlu exp/cumsum.

Output: satu file per panjang window (n=10, n=20), masing-masing berisi
lag+target keempat komponen sekaligus (biar 1 file per-n konsisten
dengan gaya log-return, bukan 8 file terpisah per komponen x window).
"""

import os
import time
from datetime import datetime

import pandas as pd
import numpy as np

print("=== CEEMDAN FEATURE ENGINEERING (WINDOWING PER-KOMPONEN) ===")
RUN_START = time.time()
RUN_START_TS = datetime.now()

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_PATH = './dataset/WTI-Expanding-LOWESS-CEEMDAN-EIA.csv'
OUTPUT_DIR = './dataset'
EVAL_DIR = './evaluations/ceemdan-eia1986-2026'
RUN_LOG_PATH = f'{EVAL_DIR}/run_log_feature_engineering.txt'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

COMPONENTS = ['Trend', 'group_1', 'group_2', 'res']
WINDOW_LENGTHS = [10, 20]  # dikonfirmasi 2026-07-06, konsisten dgn log-return

step_timings = []


def log_step(step_name, start_time):
    duration = time.time() - start_time
    step_timings.append((step_name, duration))
    print(f"   [{step_name}] selesai dalam {duration:.3f} detik")
    return time.time()


# ==========================================
# 2. LOAD DATASET
# ==========================================
t0 = time.time()
print(f"\n1. Loading dataset from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"   -> Shape: {df.shape[0]} rows x {df.shape[1]} columns "
      f"({df['Date'].min().date()} s.d. {df['Date'].max().date()})")
t0 = log_step("Load dataset", t0)


# ==========================================
# 3. WINDOWING PER KOMPONEN (SUPERVISED SEQUENCE)
# ==========================================
def build_windowed_dataset(df, n, components):
    """
    Untuk tiap target index t (mulai dari index n), bentuk lag_1..lag_n
    dan target_<komponen> untuk SETIAP komponen, plus actual_close
    (referensi evaluasi, bukan target training).
    """
    dates = df['Date'].values
    actual_close = df['actual_close'].values
    comp_arrays = {c: df[c].values for c in components}

    rows = []
    for t in range(n, len(df)):
        row = {'Date': dates[t]}
        for c in components:
            lag_window = comp_arrays[c][t - n:t]  # komponen_{t-n} ... komponen_{t-1}
            row.update({f'{c}_lag_{n - i}': lag_window[i] for i in range(n)})
            row[f'target_{c}'] = comp_arrays[c][t]
        row['actual_close'] = actual_close[t]
        rows.append(row)

    return pd.DataFrame(rows)


print("\n2. Membangun windowed dataset per-komponen (supervised lag features)...")
summary_rows = []
for n in WINDOW_LENGTHS:
    t_n = time.time()
    windowed = build_windowed_dataset(df, n, COMPONENTS)
    output_path = f'{OUTPUT_DIR}/WTI-CEEMDAN-FE-n{n}.csv'
    windowed.to_csv(output_path, index=False)
    duration = time.time() - t_n
    print(f"   -> n={n}: {windowed.shape[0]} rows x {windowed.shape[1]} columns "
          f"({n} lag x 4 komponen + 4 target + actual_close) -> {output_path} "
          f"({duration:.3f} detik)")
    summary_rows.append({
        'window_length': n,
        'n_rows': windowed.shape[0],
        'n_cols': windowed.shape[1],
        'output_path': output_path,
        'duration_sec': duration,
    })
t0 = log_step("Windowing per-komponen (n=10, n=20)", t0)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{EVAL_DIR}/feature_engineering_summary.csv', index=False)
print(f"\n   -> Ringkasan feature engineering disimpan di: "
      f"{EVAL_DIR}/feature_engineering_summary.csv")

# ==========================================
# 4. SANITY CHECK: identity Close = Trend + group_1 + group_2 + res
#    masih terjaga di baris yang sama setelah windowing (verifikasi
#    tidak ada pergeseran index/off-by-one saat membangun target)
# ==========================================
print("\n3. Sanity check identity rekonstruksi pada target kolom (n=10)...")
check_df = pd.read_csv(f'{OUTPUT_DIR}/WTI-CEEMDAN-FE-n10.csv')
recon = (check_df['target_Trend'] + check_df['target_group_1']
         + check_df['target_group_2'] + check_df['target_res'])
max_err = (recon - check_df['actual_close']).abs().max()
print(f"   -> Max identity error (target_Trend+target_group_1+target_group_2+target_res "
      f"vs actual_close): {max_err:.2e}")
t0 = log_step("Sanity check identity", t0)

# ==========================================
# 5. RUN LOG
# ==========================================
run_end_ts = datetime.now()
total_duration = time.time() - RUN_START

with open(RUN_LOG_PATH, 'a', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(f"RUN START : {RUN_START_TS.isoformat(timespec='seconds')}\n")
    f.write(f"RUN END   : {run_end_ts.isoformat(timespec='seconds')}\n")
    f.write(f"TOTAL DURATION : {total_duration:.3f} detik\n")
    f.write(f"Script : 12-ceemdan-feature-engineering.py\n")
    f.write(f"Input  : {INPUT_PATH} (shape={df.shape})\n")
    f.write("Step timings:\n")
    for step_name, duration in step_timings:
        f.write(f"  - {step_name}: {duration:.3f} detik\n")
    f.write(f"Sanity check max identity error (n=10): {max_err:.2e}\n")
    f.write("Output datasets:\n")
    f.write(summary_df.to_string(index=False) + "\n")
    f.write("\n")

print(f"\nDONE! Run log ditambahkan ke: {RUN_LOG_PATH}")
print(f"Total durasi eksekusi: {total_duration:.3f} detik")
