"""
09-ceemdan-decomposition-module.py

Tahap 2 (CEEMDAN) - langkah 3, lihat modeling-plan-ceemdan.md bagian 3.
Membangun & memvalidasi modul `decompose_residual()`: pecah Residual
(dari LOWESS, frac=0.02) jadi CEEMDAN IMF, lalu kelompokkan jadi N=2
grup tetap (dikonfirmasi user 2026-07-04) via ZCR + SampEn gabungan,
BUKAN posisi/index IMF mentah (sudah terbukti tidak stabil antar run --
lihat ceemdan-trials-comparison-test-v2.py, jumlah IMF berubah-ubah
bahkan di trials sama).

Strategi grouping (langkah 2, dikonfirmasi bersamaan dgn N=2):
  1. Hitung ZCR (zero-crossing rate, dinormalisasi) & SampEn
     (antropy.sample_entropy, order=2, metric='chebyshev') per IMF.
  2. Ranking RELATIF per window (bukan threshold absolut) -- robust utk
     expanding window yg makin panjang: rank ZCR & rank SampEn masing2,
     rata-ratakan jadi 1 composite score per IMF.
  3. Split di median composite score:
     - group_1 (noise-like) = IMF dgn composite > median
     - group_2 (trend-like) = IMF dgn composite <= median
  4. `res` (residue akhir CEEMDAN) TETAP TERPISAH dari group_1/group_2
     (ikut definisi asli plan: "Trend, group_1, ..., group_N, Res").

Validasi (SEKALI, full-series, BUKAN expanding -- expanding window baru
di langkah 5): identity Residual = group_1 + group_2 + res harus exact
(floating-point noise saja).

Parameter trials=100 (bukan 50) -- ini validasi SEKALI-JALAN, akurasi
lebih diutamakan drpd kecepatan (beda dgn loop produksi yg pakai
trials=50, lihat modeling-plan-ceemdan.md "Sudah Diputuskan").

RE-RUN 2026-07-04: dataset sumber sekarang EIA 1986-2026 (lihat
modeling-plan-eia-migration.md). Output lama (dataset 2001-2026, folder
evaluations/ceemdan/) TETAP diarsipkan, tidak ditimpa -- output baru ke
folder terpisah.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyEMD import CEEMDAN
from antropy import sample_entropy

INPUT_PATH = './dataset/WTI-LOWESS-Residual-FullSeries-EIA-frac0.02.csv'
DATASET_OUTPUT_PATH = './dataset/WTI-CEEMDAN-Groups-FullSeries-EIA.csv'
EVAL_DIR = './evaluations/ceemdan-eia1986-2026'
PLOTS_DIR = f'{EVAL_DIR}/plots'
REPORT_PATH = f'{EVAL_DIR}/ceemdan-decomposition-report.txt'
TRIALS = 100

sns.set_theme(style="whitegrid")


def zcr(imf):
    zero_crosses = np.nonzero(np.diff(np.sign(imf)))[0]
    return len(zero_crosses) / len(imf)


def decompose_residual(residual_series, trials=TRIALS):
    """Modul reusable: pecah Residual jadi group_1 (noise-like),
    group_2 (trend-like), res (residue akhir CEEMDAN).
    Return dict + info diagnostik per-IMF (utk laporan/plot)."""
    # parallel=False eksplisit -- lihat catatan bug WinError 1450 di
    # 10-expanding-window-decomposition.py (default PyEMD parallel=True
    # diam-diam bikin multiprocessing.Pool, bisa habiskan handle IPC Windows)
    imfs = CEEMDAN(trials=trials, parallel=False)(residual_series)
    imf_rows = imfs[:-1]
    res = imfs[-1]
    n_imf = imf_rows.shape[0]

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

    diagnostics = pd.DataFrame({
        'imf_index': np.arange(1, n_imf + 1),
        'zcr': zcrs,
        'sample_entropy': sampens,
        'rank_zcr': rank_zcr,
        'rank_sampen': rank_sampen,
        'composite_score': composite,
        'group': np.where(noise_mask, 'group_1_noise', 'group_2_trend'),
    })

    return {
        'group_1': group_1,
        'group_2': group_2,
        'res': res,
        'n_imf': n_imf,
        'diagnostics': diagnostics,
        'raw_imfs': imf_rows,
    }


def main():
    print("=== VALIDASI MODUL DEKOMPOSISI CEEMDAN (N=2 GRUP, FULL-SERIES) ===")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=['Date'])
    residual = df['Residual'].to_numpy()
    print(f"Input: {INPUT_PATH} ({len(residual)} baris)")
    print(f"trials={TRIALS} (validasi sekali-jalan, akurasi diutamakan)\n")

    result = decompose_residual(residual, trials=TRIALS)
    print(f"n_IMF dihasilkan: {result['n_imf']}")
    print(result['diagnostics'].to_string(index=False))

    # ==========================================
    # VALIDASI IDENTITY: Residual = group_1 + group_2 + res
    # ==========================================
    reconstructed = result['group_1'] + result['group_2'] + result['res']
    identity_error = np.abs(residual - reconstructed).max()
    print(f"\nIdentity check (Residual = group_1 + group_2 + res): "
          f"max error = {identity_error:.2e}")
    assert identity_error < 1e-6, "Identity check GAGAL -- cek ulang modul!"
    print("-> IDENTITY CHECK LOLOS.")

    # ==========================================
    # SIMPAN DATASET (full-series, BUKAN training final)
    # ==========================================
    out_df = df.copy()
    out_df['group_1'] = result['group_1']
    out_df['group_2'] = result['group_2']
    out_df['res'] = result['res']
    out_df.to_csv(DATASET_OUTPUT_PATH, index=False)
    print(f"\nDataset (full-series, utk validasi/plot saja) disimpan: {DATASET_OUTPUT_PATH}")

    # Sanity check tambahan: Close = Trend + group_1 + group_2 + res
    full_reconstructed = out_df['Trend'] + out_df['group_1'] + out_df['group_2'] + out_df['res']
    full_identity_error = (out_df['Close'] - full_reconstructed).abs().max()
    print(f"Sanity check tambahan (Close = Trend + group_1 + group_2 + res): "
          f"max error = {full_identity_error:.2e}")

    # ==========================================
    # PLOT: stacked IMF spectrum + grup
    # ==========================================
    print("\nMembuat plot...")
    n_imf = result['n_imf']
    raw_imfs = result['raw_imfs']
    diag = result['diagnostics']

    fig, axes = plt.subplots(n_imf + 1, 1, figsize=(15, 2 * (n_imf + 1)), sharex=True)
    dates = df['Date']
    for i in range(n_imf):
        ax = axes[i]
        group_label = diag.loc[i, 'group']
        color = 'salmon' if group_label == 'group_1_noise' else 'steelblue'
        ax.plot(dates, raw_imfs[i], color=color, linewidth=1)
        ax.set_ylabel(f"IMF {i+1}\n({group_label})", fontsize=9, rotation=0,
                      labelpad=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.3)
    axes[-1].plot(dates, result['res'], color='black', linewidth=1)
    axes[-1].set_ylabel('res', fontsize=9, rotation=0, labelpad=45, ha='right')
    axes[-1].grid(True, linestyle='--', alpha=0.3)
    axes[-1].set_xlabel('Date')
    fig.suptitle('CEEMDAN IMF Spectrum -- Warna = Grup (salmon=noise, biru=trend-like)',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(f'{PLOTS_DIR}/01_ceemdan_imf_spectrum_grouped.png', dpi=200)
    plt.close(fig)

    # Plot 2: SEMUA IMF mentah (individual, bukan per grup) ditumpuk jadi
    # SATU plot -- beda dari plot 1 (terpisah per subplot) dan plot 3
    # (sudah dijumlah per grup). Warna dari colormap supaya tiap IMF
    # bisa dibedakan.
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = sns.color_palette('viridis', n_colors=n_imf)
    for i in range(n_imf):
        group_label = diag.loc[i, 'group']
        linestyle = '-' if group_label == 'group_1_noise' else '--'
        ax.plot(dates, raw_imfs[i], color=colors[i], linewidth=1, linestyle=linestyle,
                label=f"IMF {i+1} ({group_label.replace('group_1_noise', 'noise').replace('group_2_trend', 'trend')})",
                alpha=0.8)
    ax.plot(dates, result['res'], color='black', linewidth=1.5, label='res')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('Semua IMF Mentah Ditumpuk dalam Satu Plot (garis solid=group_1 noise, '
                 'putus-putus=group_2 trend)')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/02_ceemdan_all_imf_overlay.png', dpi=200)
    plt.close(fig)

    # Plot 3: hasil GRUP (sudah dijumlah, group_1 + group_2 + res) vs Residual asli.
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, out_df['group_1'], label='group_1 (noise-like)', color='salmon', alpha=0.8)
    ax.plot(dates, out_df['group_2'], label='group_2 (trend-like)', color='steelblue', alpha=0.8)
    ax.plot(dates, out_df['res'], label='res', color='black', alpha=0.8)
    ax.plot(dates, out_df['Residual'], label='Residual asli (LOWESS)', color='gray',
            linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('Grup Hasil CEEMDAN vs Residual Asli')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/03_ceemdan_groups_vs_residual.png', dpi=200)
    plt.close(fig)

    # Plot 3: ZCR vs SampEn per IMF (dual-axis bar chart, gaya
    # previous-03a-imf-justification.py) -- justifikasi visual kenapa
    # IMF tertentu masuk group_1 (noise) vs group_2 (trend).
    labels = [f'IMF {i}' for i in diag['imf_index']]
    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x_pos - width / 2, diag['sample_entropy'], width,
            label='Sample Entropy', color='darkred', alpha=0.8)
    ax1.set_xlabel('Intrinsic Mode Functions (IMF)')
    ax1.set_ylabel('Sample Entropy Value', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.bar(x_pos + width / 2, diag['zcr'], width, label='ZCR', color='teal', alpha=0.8)
    ax2.set_ylabel('Zero Crossing Rate (ZCR)', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')

    # Tandai garis pemisah noise_group vs trend_group di median composite score
    n_noise = (diag['group'] == 'group_1_noise').sum()
    if 0 < n_noise < len(diag):
        ax1.axvline(n_noise - 0.5, color='black', linestyle='--', linewidth=1.5,
                    label='Batas group_1/group_2')

    plt.title('Complexity Analysis: Sample Entropy vs ZCR per IMF\n'
              '(garis putus-putus = batas group_1 noise-like / group_2 trend-like)',
              fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/04_entropy_zcr_comparison.png', dpi=200)
    plt.close(fig)

    print(f"Plot disimpan di: {PLOTS_DIR}")

    # ==========================================
    # LAPORAN
    # ==========================================
    lines = [
        "CEEMDAN DECOMPOSITION MODULE VALIDATION REPORT",
        "=" * 70,
        f"Input: {INPUT_PATH} ({len(residual)} baris)",
        f"trials = {TRIALS}",
        f"n_IMF dihasilkan: {n_imf}",
        "",
        "Strategi grouping: ranking relatif (composite score = rata-rata rank ZCR "
        "+ rank SampEn), split di median -> N=2 grup tetap (group_1=noise-like, "
        "group_2=trend-like). res (residue akhir CEEMDAN) tetap terpisah.",
        "",
        "Detail per-IMF:",
        result['diagnostics'].to_string(index=False),
        "",
        f"Identity check (Residual = group_1 + group_2 + res): "
        f"max error = {identity_error:.2e} -> {'LOLOS' if identity_error < 1e-6 else 'GAGAL'}",
        f"Sanity check (Close = Trend + group_1 + group_2 + res): "
        f"max error = {full_identity_error:.2e}",
        "",
        "PENTING: ini dataset FULL-SERIES (bukan expanding window) -- HANYA untuk "
        "validasi modul benar secara matematis. Dataset training final baru ada "
        "setelah expanding-window engine (langkah 5) dibangun & dijalankan.",
    ]
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nLaporan disimpan: {REPORT_PATH}")


if __name__ == '__main__':
    main()
