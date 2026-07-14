# Laporan Presentasi Final

Laporan ini disusun bertahap sesuai arahan, urutan dan isi tiap bagian ditentukan menyusul.

## 1. LOWESS (Locally Weighted Scatterplot Smoothing)

### 1.1. Konsep dan Alasan Pemilihan

Harga penutupan WTI terbukti non stasioner pada uji eksploratori awal.
Trennya perlu dihilangkan terlebih dahulu sebelum sisa series bisa dianalisis atau dimodelkan dengan wajar.

LOWESS dipilih untuk mengekstrak komponen tren jangka panjang yang smooth, bukan untuk menstasionerkan lewat transformasi matematis seperti differencing atau log return.
Definisi dekomposisi tahap ini sebagai berikut.

```
Close = Trend (LOWESS) + Residual
```

Residual hasil pengurangan inilah yang selanjutnya diproses CEEMDAN untuk dipecah lagi menjadi osilasi osilasi (IMF) di sekitar nol.
Komponen Trend inilah yang menjadi salah satu dari 4 fitur input model deep learning pada proyek ini (Trend, IMF Group 1, IMF Group 2, Residual).

### 1.2. Kode Implementasi (Skema Produksi, Expanding Window)

Sumber, `D:\Coding\#bigdata\crude-oil-forecasting-thesis\10-expanding-window-decomposition.py`, script inilah yang secara aktual menghasilkan dataset final proyek ini (`WTI-CEEMDAN-FE-n10.csv`, lewat tahapan lanjutan `12-ceemdan-feature-engineering.py`).
Salinan lengkap script disimpan di `references/scripts/10-expanding-window-decomposition.py`.

Untuk setiap titik waktu t, prosedurnya sebagai berikut.

Ambil Close dari titik pertama sampai t (expanding window, tidak pernah melihat data masa depan).
Jalankan LOWESS pada window itu, Trend pada titik t diambil dari elemen terakhir hasil smoothing (endpoint).
Residual seluruh window (bukan hanya titik t) menjadi input tahap CEEMDAN berikutnya.

Fungsi inti ekstraksi Trend dan Residual.

```python
FRAC = 0.02          # dikonfirmasi final, modeling-plan-lowess.md
WARMUP_MINIMUM = 500  # baris pertama dipakai warmup, di-drop dari dataset final

def extract_trend_and_residual(close_window):
    """Modul LOWESS, return Trend_t (endpoint) dan
    Residual_window (seluruh kurva, dipakai sbg input CEEMDAN)."""
    x = np.arange(len(close_window))
    smoothed = lowess(close_window, x, frac=FRAC, return_sorted=False)
    trend_t = smoothed[-1]
    residual_window = close_window - smoothed
    return trend_t, residual_window
```

Loop produksi memanggil fungsi ini pada setiap titik waktu t, mulai dari t = 500 (`WARMUP_MINIMUM`) sampai baris terakhir dataset.
Baris sebelum t = 500 tidak memiliki window yang cukup panjang untuk menghasilkan estimasi Trend endpoint yang layak, sehingga di-drop dari dataset final.
Ini adalah alasan dataset final proyek ini dimulai dari 1988-01-11, bukan dari awal data harga minyak mentah pada 1986-01-02.

Reproducibility dijaga dengan seed CEEMDAN dikunci per hari (`seed = SEED_BASE + t`), sehingga pipeline ini menghasilkan angka yang sama persis setiap kali dijalankan ulang dari awal.
Estimasi runtime penuh pipeline (LOWESS dan CEEMDAN gabungan, 50 trial per hari, seluruh dataset 1986 sampai 2026) sekitar 31 jam, sehingga script dilengkapi mekanisme checkpoint dan resume otomatis.

### 1.3. Pemilihan Parameter Frac

Parameter utama LOWESS adalah `frac`, proporsi data yang dipakai sebagai tetangga lokal di tiap titik smoothing (`statsmodels.nonparametric.smoothers_lowess.lowess`).

Trade off pemilihan frac sebagai berikut.
Frac terlalu kecil membuat Trend terlalu mengikuti noise jangka pendek (overfitting), Residual kehilangan sinyal yang seharusnya menjadi bagian Trend.
Frac terlalu besar membuat Trend terlalu halus (underfitting), sisa tren jangka panjang malah bocor ke Residual, membuat Residual masih non stasioner dan tugas CEEMDAN berikutnya lebih berat.

Tujuh kandidat frac diuji secara lengkap (0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2) pada dekomposisi full series (bukan expanding window, khusus untuk validasi pemilihan parameter) memakai dataset EIA 1986 sampai 2026 (10.191 baris), identik dengan dataset sumber proyek ini.
Ringkasan hasilnya sebagai berikut, lengkap di `evaluations/statistical/lowess/lowess-decomposition-report.txt`.

| Frac | Residual std | Residual skew | Residual kurtosis | ADF p value residual |
|---|---:|---:|---:|---:|
| 0.01 | 3.4309 | -0.6874 | 26.7406 | 1.5486e-27 |
| 0.02 | 5.4101 | -0.5652 | 15.6649 | 3.1788e-24 |
| 0.03 | 6.6564 | 0.1164 | 12.2461 | 3.7186e-21 |
| 0.04 | 7.7365 | 0.4992 | 11.1689 | 1.0955e-18 |
| 0.05 | 8.5358 | 0.8683 | 10.5306 | 1.6989e-16 |
| 0.10 | 11.2693 | 1.2728 | 7.9681 | 1.0304e-10 |
| 0.20 | 12.6831 | 0.7856 | 4.7996 | 2.1592e-08 |

Seluruh kandidat frac lolos uji stasioneritas ADF dan KPSS pada Residual, sehingga pemilihan akhir tidak ditentukan oleh stasioneritas melainkan oleh dua pertimbangan lain, kualitas smoothing (kurtosis Residual) dan stabilitas endpoint (dijelaskan pada Bagian 1.4).

Kurtosis Residual turun tajam dari frac 0.01 ke 0.02 (26.74 menjadi 15.66), lalu melandai bertahap untuk frac yang lebih besar.
Penurunan tajam ini mengindikasikan frac 0.01 masih terlalu menempel pada Close (Trend terlalu tajam mengikuti noise), sementara frac 0.02 sudah menunjukkan perbaikan signifikan.

### 1.4. Isu Teknis Boundary Bias pada Expanding Window

LOWESS menghitung regresi lokal berbobot di tiap titik berdasarkan tetangga di kedua sisi, kiri dan kanan.
Pada titik paling akhir sebuah window, yaitu titik yang justru paling dibutuhkan untuk forecasting (t saat ini), tidak ada tetangga di sisi kanan sama sekali.
Estimasi Trend di endpoint ini secara statistik dikenal kurang stabil, disebut boundary bias atau edge effect.

Validasi dilakukan dengan membandingkan Trend di suatu titik cutoff yang dihitung sebagai endpoint window (kondisi asli forecasting, tanpa data masa depan) terhadap Trend di titik yang sama setelah dihitung ulang dengan window diperpanjang k hari ke depan (titik itu menjadi titik interior, punya tetangga kanan).
Sembilan tanggal cutoff diuji, mencakup berbagai rezim pasar (pasca crash 1986, pasca Perang Teluk 1991, sebelum krisis finansial 2008, tengah crash minyak 2014 2016, dan seterusnya), dengan perpanjangan k sebesar 1, 5, 10, 20, 60, dan 120 hari bursa.

Metrik paling relevan adalah pergeseran pada k=1, karena loop produksi expanding window hanya maju 1 hari per langkah dan tidak pernah menghitung ulang Trend masa lalu.
Ringkasan agregat lintas 9 titik cutoff pada metrik k=1 ini sebagai berikut, lengkap di `evaluations/statistical/lowess/boundary-bias-report.txt`.

| Frac | Pergeseran maksimum pada k=1 | Persentase kombinasi signifikan (lebih dari 2 persen) pada k=1 |
|---|---:|---:|
| 0.01 | 1.233% | 0% |
| 0.02 | 0.688% (terendah) | 0% |
| 0.03 | 2.292% | 11.1% (1 dari 9 cutoff) |
| 0.04 | 1.458% | 0% |
| 0.05 | 1.601% | 0% |
| 0.10 | 2.944% | 11.1% |
| 0.20 | 1.540% | 0% |

Frac 0.02 menunjukkan pergeseran paling rendah pada metrik k=1 (0,688 persen), bahkan lebih stabil dari frac 0.01, dan tidak ada kombinasi yang melewati ambang signifikan 2 persen pada frac ini.

Pada metrik pergeseran maksimum lintas seluruh rentang k (1 sampai 120 hari), hasilnya jauh lebih berisik untuk semua frac, termasuk frac 0.02 yang bisa mencapai 21,8 persen.
Namun pergeseran besar ini didominasi oleh satu kombinasi outlier, cutoff 2015-01-02 pada k=20 hari, tepat di tengah crash harga minyak 2014 2016.
Temuan ini konsisten dengan analisis terpisah bahwa rezim crash tertentu memang secara struktural tidak stabil untuk LOWESS pada frac berapapun, dan telah dibuktikan tidak dapat diperbaiki lewat berbagai upaya mitigasi (pengujian additive outlier modeling dan tuning iterasi robust reweighting, keduanya gagal mengurangi pergeseran secara berarti).
Karena metrik k=1 adalah satu satunya yang relevan dengan mekanisme produksi aktual, temuan pada k besar ini tidak mengubah keputusan pemilihan frac.

### 1.5. Keputusan Final

Frac 0.02 dikonfirmasi sebagai parameter final, dikuatkan oleh validasi pada dataset EIA 1986 sampai 2026 yang mencakup rentang volatilitas jauh lebih panjang dari validasi awal.
Pertimbangan utamanya sebagai berikut.

Stabilitas endpoint pada metrik k=1 (paling relevan dengan mekanisme produksi expanding window), frac 0.02 adalah yang terbaik dari seluruh kandidat.
Kualitas smoothing sudah membaik signifikan dibanding frac 0.01 (kurtosis Residual turun dari 26,74 menjadi 15,66), meski belum sehalus frac yang lebih besar.

Boundary bias pada rezim pasar dengan structural break ekstrem (seperti crash minyak 2014 2016) diterima sebagai keterbatasan struktural LOWESS yang terdokumentasi, bukan sesuatu yang dimitigasi lewat algoritma tambahan di pipeline produksi, karena seluruh upaya mitigasi yang diuji terbukti tidak efektif.

### 1.6. Visualisasi

Perbandingan kurva Trend untuk seluruh kandidat frac pada dekomposisi full series (`evaluations/graphical/lowess/01_trend_overlay_all_frac.png`).

Sebaran waktu dan histogram Residual pada frac 0.02, parameter final (`evaluations/graphical/lowess/02_residual_timeseries_frac_0.02.png` dan `evaluations/graphical/lowess/03_residual_histogram_frac_0.02.png`).

Contoh validasi boundary bias pada titik cutoff yang relatif stabil, 1987-01-02 pasca crash harga minyak 1986 (`evaluations/graphical/lowess/04_boundary_bias_cutoff_19870102.png`), dan pada titik cutoff paling tidak stabil, 2015-01-02 tengah crash harga minyak 2014 2016 (`evaluations/graphical/lowess/04_boundary_bias_cutoff_20150102.png`), menunjukkan kontras antara rezim pasar stabil dan rezim structural break.
