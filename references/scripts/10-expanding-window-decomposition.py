"""
10-expanding-window-decomposition.py

Tahap 2 (CEEMDAN) - langkah 5, lihat modeling-plan-ceemdan.md bagian 5.
(Penomoran: plan awal menyebut "09-expanding-window-decomposition.py",
tapi 09 sudah dipakai 09-ceemdan-decomposition-module.py, jadi ini 10.)

INI SCRIPT PALING BERAT & PALING KRITIS di seluruh pipeline. Menyatukan
LOWESS (extract_trend, frac=0.02) dan CEEMDAN (decompose_residual, N=2
grup via ZCR+SampEn ranking relatif, trials=50) jadi SATU LOOP HARIAN
atomik, expanding window (tidak pernah melihat data masa depan).

Untuk setiap titik waktu t:
  1. Ambil Close[0...t] (expanding window, dari titik pertama sampai t).
  2. Jalankan LOWESS di window itu -> Trend_t = smoothed[-1] (endpoint),
     Residual_window = Close[0...t] - smoothed (SELURUH kurva window,
     bukan cuma titik t -- residual window ini yang jadi input CEEMDAN).
  3. Jalankan CEEMDAN di Residual_window (SELURUH window) -> IMF utk
     seluruh window -> kelompokkan ZCR+SampEn (ranking relatif thd
     window itu sendiri) jadi group_1/group_2/res -> ambil ENDPOINT
     (elemen terakhir) dari masing2 sbg group_1_t/group_2_t/res_t.
  4. Simpan satu baris: Date, Trend_t, group_1_t, group_2_t, res_t,
     actual_close.

ESTIMASI RUNTIME: ~31 jam (trials=50, dataset EIA 1986-2026 ~10191 baris,
lihat evaluations/ceemdan-eia1986-2026/summary-ceemdan-runtime-benchmark.txt
dan modeling-plan-eia-migration.md untuk detail benchmark & kalkulasi --
angka ini KOREKSI dari estimasi lama ~18-19 jam, yang ternyata dihitung
dari benchmark yang diam-diam parallel, bukan single-process murni).
Estimasi ini dari biaya CEEMDAN mentah saja -- overhead ZCR+SampEn+grouping
per hari menambah sedikit lagi (kemungkinan tidak signifikan dibanding
CEEMDAN, tapi belum diukur terpisah).

CHECKPOINTING: WAJIB krn runtime sangat panjang. Script ini RESUMABLE --
kalau dihentikan (Ctrl+C, crash, restart PC), jalankan ulang script yang
SAMA, otomatis lanjut dari baris terakhir yang sudah tersimpan di
OUTPUT_PATH, bukan mulai dari nol. Progress disimpan setiap
CHECKPOINT_INTERVAL baris.

REPRODUCIBILITY: CEEMDAN py stokastik (~21% variasi run-ke-run terbukti
di ceemdan-trials-comparison-test-v2.py). Seed per-hari DIKUNCI
(seed = SEED_BASE + t) supaya pipeline ini bisa diulang dan menghasilkan
angka SAMA PERSIS setiap kali dijalankan dari awal -- bukan acak beda
tiap run.

WARMUP_MINIMUM: PLACEHOLDER (500 baris, sama dgn window terkecil yang
diuji di benchmark) -- keputusan "window minimum sebelum Trend/Residual
endpoint layak dipakai" (modeling-plan-lowess.md) SENGAJA ditunda
(instruksi user). Kalau nanti nilai minimum yang lebih tinggi diputuskan,
TIDAK PERLU rerun apa pun -- tinggal filter/buang baris-baris awal dari
dataset output ini (tiap baris independen scr komputasi, cuma soal mana
yang layak dipakai training).

BUG KRITIS DITEMUKAN & DIPERBAIKI (2026-07-04, dari crash di
08-ceemdan-runtime-benchmark.py): `CEEMDAN(trials=TRIALS)` TANPA
`parallel=False` eksplisit diam-diam memakai multiprocessing.Pool
(default PyEMD `parallel=True`). Karena `decompose_residual()` di sini
membuat instance CEEMDAN baru SETIAP HARI (~9700 kali untuk seluruh
dataset EIA), Pool() akan dibuat berulang ribuan kali sepanjang run
~37 jam ini -- di Windows ini akan HAMPIR PASTI crash di tengah jalan
dengan `OSError: WinError 1450` (kehabisan named-pipe handle utk IPC)
begitu resource itu terkuras, sama seperti yang terjadi di benchmark.
Diperbaiki dengan `parallel=False` eksplisit -- run produksi ini TIDAK
PERNAH membuat Pool() sama sekali, menghilangkan risiko crash ini
sepenuhnya (bukan cuma mengurangi peluangnya).
"""

import os
import time

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from PyEMD import CEEMDAN
from antropy import sample_entropy

INPUT_PATH = './dataset/WTI-EIAGOV-1986-2026.csv'
OUTPUT_PATH = './dataset/WTI-Expanding-LOWESS-CEEMDAN-EIA.csv'
EVAL_DIR = './evaluations/ceemdan-eia1986-2026'
LOG_PATH = f'{EVAL_DIR}/expanding_window_run_log.txt'

FRAC = 0.02          # dikonfirmasi final, modeling-plan-lowess.md
TRIALS = 50          # dikonfirmasi final utk loop produksi, modeling-plan-ceemdan.md
SEED_BASE = 1000      # seed per-hari = SEED_BASE + t, utk reproducibility
WARMUP_MINIMUM = 500  # PLACEHOLDER, lihat docstring
CHECKPOINT_INTERVAL = 50


def zcr(imf):
    zero_crosses = np.nonzero(np.diff(np.sign(imf)))[0]
    return len(zero_crosses) / len(imf)


def extract_trend_and_residual(close_window):
    """Modul LOWESS (sama dgn 06/07): return Trend_t (endpoint) dan
    Residual_window (seluruh kurva, dipakai sbg input CEEMDAN)."""
    x = np.arange(len(close_window))
    smoothed = lowess(close_window, x, frac=FRAC, return_sorted=False)
    trend_t = smoothed[-1]
    residual_window = close_window - smoothed
    return trend_t, residual_window


def decompose_residual(residual_window, seed):
    """Modul CEEMDAN (sama dgn 09): N=2 grup via ZCR+SampEn ranking
    relatif. Return group_1_t, group_2_t, res_t (ENDPOINT/elemen
    terakhir tiap komponen)."""
    ceemdan = CEEMDAN(trials=TRIALS, parallel=False)
    ceemdan.noise_seed(seed)
    imfs = ceemdan(residual_window)
    imf_rows = imfs[:-1]
    res = imfs[-1]
    n_imf = imf_rows.shape[0]

    if n_imf == 0:
        return 0.0, 0.0, res[-1]

    zcrs = np.array([zcr(imf) for imf in imf_rows])
    sampens = np.array([
        sample_entropy(imf, order=2, metric='chebyshev') for imf in imf_rows
    ])
    rank_zcr = pd.Series(zcrs).rank().to_numpy()
    rank_sampen = pd.Series(sampens).rank().to_numpy()
    composite = (rank_zcr + rank_sampen) / 2
    median_composite = np.median(composite)

    noise_mask = composite > median_composite
    trend_mask = ~noise_mask

    group_1 = imf_rows[noise_mask].sum(axis=0) if noise_mask.any() else np.zeros_like(res)
    group_2 = imf_rows[trend_mask].sum(axis=0) if trend_mask.any() else np.zeros_like(res)

    return group_1[-1], group_2[-1], res[-1]


def main():
    print("=== EXPANDING-WINDOW LOWESS+CEEMDAN ENGINE ===")
    os.makedirs(EVAL_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
    df = df.rename(columns={'Price': 'Close'})
    df = df.sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)
    close = df['Close'].to_numpy()
    dates = df['Date'].to_numpy()
    n_total = len(close)
    print(f"Total baris dataset: {n_total}")
    print(f"FRAC={FRAC}, TRIALS={TRIALS}, WARMUP_MINIMUM={WARMUP_MINIMUM} (placeholder)\n")

    # ==========================================
    # RESUME LOGIC: cek progress yang sudah ada
    # ==========================================
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        n_done = len(existing_df)
        start_t = WARMUP_MINIMUM + n_done
        print(f"Ditemukan progress sebelumnya: {n_done} baris sudah selesai.")
        print(f"Melanjutkan dari t={start_t} ({df.loc[start_t, 'Date'].date() if start_t < n_total else 'SELESAI'})\n")
        file_mode = 'a'
        write_header = False
    else:
        start_t = WARMUP_MINIMUM
        print(f"Belum ada progress. Mulai dari t={start_t} ({df.loc[start_t, 'Date'].date()})\n")
        file_mode = 'w'
        write_header = True

    if start_t >= n_total:
        print("Semua baris sudah selesai diproses. Tidak ada yang perlu dikerjakan.")
        return

    buffer = []
    run_start_time = time.time()
    durations = []

    print("Tekan Ctrl+C kapan saja untuk PAUSE -- progress sampai titik terakhir yang "
          "selesai akan otomatis disimpan sebelum keluar. Jalankan ulang script yang "
          "sama untuk RESUME dari titik itu.\n")

    def flush_buffer(buf, mode, header):
        """Simpan buffer ke OUTPUT_PATH walau belum penuh CHECKPOINT_INTERVAL --
        dipakai baik oleh checkpoint normal maupun oleh pause (Ctrl+C)."""
        if buf:
            pd.DataFrame(buf).to_csv(OUTPUT_PATH, mode=mode, header=header, index=False)
        return 'a', False

    last_completed_t = start_t - 1

    try:
        for t in range(start_t, n_total):
            t0 = time.time()

            close_window = close[0:t + 1]
            trend_t, residual_window = extract_trend_and_residual(close_window)
            group_1_t, group_2_t, res_t = decompose_residual(residual_window, seed=SEED_BASE + t)

            buffer.append({
                'Date': pd.Timestamp(dates[t]).date().isoformat(),
                'Trend': trend_t,
                'group_1': group_1_t,
                'group_2': group_2_t,
                'res': res_t,
                'actual_close': close[t],
            })
            last_completed_t = t  # Ctrl+C sesudah baris ini -> titik ini AMAN tersimpan

            duration = time.time() - t0
            durations.append(duration)

            # ==========================================
            # LIVE PROGRESS (tiap baris, satu baris yang di-overwrite --
            # supaya terminal tidak diam/blinking selama puluhan detik per titik)
            # ==========================================
            avg_recent = np.mean(durations[-50:])
            n_remaining_live = n_total - 1 - t
            eta_live_sec = avg_recent * n_remaining_live
            elapsed_live = time.time() - run_start_time
            print(f"\r[{pd.Timestamp.now().strftime('%H:%M:%S')}] t={t}/{n_total-1} "
                  f"({pd.Timestamp(dates[t]).date()}) | titik ini={duration:.1f}s | "
                  f"avg50={avg_recent:.1f}s/titik | elapsed={elapsed_live/3600:.2f}h | "
                  f"ETA sisa={eta_live_sec/3600:.2f}h    ", end='', flush=True)

            # ==========================================
            # CHECKPOINT
            # ==========================================
            is_last = (t == n_total - 1)
            if len(buffer) >= CHECKPOINT_INTERVAL or is_last:
                file_mode, write_header = flush_buffer(buffer, file_mode, write_header)
                buffer = []

                avg_duration = np.mean(durations[-500:])  # rata-rata 500 titik terakhir
                n_remaining = n_total - 1 - t
                eta_sec = avg_duration * n_remaining
                elapsed = time.time() - run_start_time
                print(f"\n[Checkpoint] t={t}/{n_total-1} ({pd.Timestamp(dates[t]).date()}) | "
                      f"window={t+1} baris | avg={avg_duration:.2f}s/titik | "
                      f"elapsed={elapsed/3600:.2f}h | ETA sisa={eta_sec/3600:.2f}h")

                with open(LOG_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{pd.Timestamp.now().isoformat()} | t={t} | "
                            f"avg={avg_duration:.2f}s/titik | ETA={eta_sec/3600:.2f}h\n")

        total_duration = time.time() - run_start_time
        print(f"\n=== SELESAI. Total durasi: {total_duration/3600:.2f} jam ===")
        print(f"Dataset disimpan: {OUTPUT_PATH}")

    except KeyboardInterrupt:
        # ==========================================
        # PAUSE (Ctrl+C): flush sisa buffer yang belum sempat checkpoint
        # normal, lalu keluar bersih -- resume otomatis lewat logic di atas
        # begitu script ini dijalankan ulang.
        # ==========================================
        n_flushed = len(buffer)
        file_mode, write_header = flush_buffer(buffer, file_mode, write_header)
        elapsed = time.time() - run_start_time
        print("\n\n=== DIPAUSE (Ctrl+C) ===")
        if last_completed_t >= start_t:
            print(f"Progress tersimpan sampai t={last_completed_t} "
                  f"({pd.Timestamp(dates[last_completed_t]).date()}) -- {n_flushed} baris "
                  f"terakhir baru saja di-flush ke disk.")
        else:
            print("Belum ada titik baru yang selesai di sesi ini (interrupt terjadi sebelum "
                  "titik pertama tuntas) -- progress tetap sama seperti sebelum run ini.")
        print(f"Waktu berjalan sesi ini: {elapsed/3600:.2f} jam.")
        print(f"Untuk lanjut: jalankan ulang script ini (python 10-expanding-window-"
              f"decomposition.py) -- otomatis resume dari t={last_completed_t + 1}.")
        return


if __name__ == '__main__':
    main()
