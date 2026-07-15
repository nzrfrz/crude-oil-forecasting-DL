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

Catatan penting, `WARMUP_MINIMUM` tidak dipakai di dalam fungsi `extract_trend_and_residual` di atas.
Fungsi ini murni menerima `close_window` apa pun yang dikirim kepadanya, tanpa mengetahui soal warmup sama sekali.

`WARMUP_MINIMUM` dipakai di fungsi `main()`, untuk menentukan titik awal `t` pada loop produksi (`start_t`), sekaligus logika resume kalau proses sempat dihentikan dan dilanjutkan.

```python
if os.path.exists(OUTPUT_PATH):
    existing_df = pd.read_csv(OUTPUT_PATH)
    n_done = len(existing_df)
    start_t = WARMUP_MINIMUM + n_done   # resume, lanjut dari progress terakhir
else:
    start_t = WARMUP_MINIMUM             # run baru, mulai dari t = 500

for t in range(start_t, n_total):
    close_window = close[0:t + 1]
    trend_t, residual_window = extract_trend_and_residual(close_window)
    ...
```

Loop produksi memanggil `extract_trend_and_residual` pada setiap titik waktu t, mulai dari t = 500 (`WARMUP_MINIMUM`) sampai baris terakhir dataset, sesuai `start_t` yang ditentukan di atas.
Baris sebelum t = 500 tidak memiliki window yang cukup panjang untuk menghasilkan estimasi Trend endpoint yang layak, sehingga tidak pernah masuk ke loop dan otomatis tidak ada di dataset final, bukan karena ada pengecekan warmup di dalam fungsi ekstraksinya.
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

## 2. CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)

### 2.1. Konsep dan Alasan Pemilihan

Residual hasil LOWESS (Bagian 1) masih berupa sinyal campuran, ada osilasi cepat (noise harian) dan osilasi lambat (siklus jangka menengah) bercampur dalam satu series.
CEEMDAN dipakai untuk memecah Residual ini menjadi beberapa Intrinsic Mode Function (IMF), tiap IMF mewakili satu skala osilasi, mulai dari frekuensi tinggi (IMF pertama) sampai frekuensi rendah (IMF terakhir), ditambah satu komponen residue akhir yang tersisa setelah semua IMF diekstrak.

CEEMDAN adalah varian Empirical Mode Decomposition (EMD) yang menambahkan noise putih terkontrol secara berulang (ensemble) untuk mengatasi masalah mode mixing pada EMD biasa, satu IMF EMD biasa bisa berisi campuran skala osilasi yang seharusnya terpisah.
Definisi dekomposisi tahap ini sebagai berikut.

```
Residual (LOWESS) = IMF_1 + IMF_2 + ... + IMF_n + res
```

Masalah praktis, jumlah IMF yang dihasilkan CEEMDAN tidak konsisten, bisa berbeda beda tergantung isi window dan sifat stokastik algoritmanya sendiri (dibahas Bagian 2.3).
Kalau tiap IMF langsung dipakai sebagai fitur terpisah, jumlah kolom fitur akan berubah ubah antar titik waktu pada skema expanding window, tidak bisa dipakai model dengan input berdimensi tetap.
Solusinya, IMF dikelompokkan menjadi jumlah grup yang TETAP (N=2), berdasarkan karakteristik osilasinya, bukan berdasarkan urutan indeksnya.

```
Residual (LOWESS) = group_1 (noise-like) + group_2 (trend-like) + res
```

Ketiga komponen ini, ditambah Trend dari LOWESS, menjadi 4 fitur input model deep learning proyek ini (Trend, group_1, group_2, res, sesuai penamaan `FEATURE_NAMES` pada `05-dl-model-training.py` dan `09-xai-explainability.py`).

### 2.2. Kode Implementasi (Modul Grouping ZCR + SampEn)

Sumber, `D:\Coding\#bigdata\crude-oil-forecasting-thesis\09-ceemdan-decomposition-module.py` (validasi modul, full series, sekali jalan) dan `10-expanding-window-decomposition.py` (loop produksi harian, sudah dibahas Bagian 1.2, satu loop atomik menggabungkan LOWESS dan CEEMDAN sekaligus).
Salinan lengkap modul validasi disimpan di `references/scripts/09-ceemdan-decomposition-module.py`.

Setiap IMF diberi skor komposit dari dua metrik komplementer.

ZCR (Zero Crossing Rate), seberapa sering sinyal berganti tanda, sinyal frekuensi tinggi (noise) menyeberangi nol lebih sering daripada sinyal frekuensi rendah (tren).
SampEn (Sample Entropy, `antropy.sample_entropy`), seberapa tidak teratur/tidak dapat diprediksi sebuah sinyal, sinyal noise punya entropi lebih tinggi daripada sinyal tren yang lebih halus.

Kedua metrik dirangking secara relatif terhadap window itu sendiri (bukan ambang batas absolut), lalu dirata rata jadi satu skor komposit per IMF, dan dibelah di titik median.

```python
def zcr(imf):
    zero_crosses = np.nonzero(np.diff(np.sign(imf)))[0]
    return len(zero_crosses) / len(imf)


def decompose_residual(residual_series, trials=TRIALS):
    """Modul reusable, pecah Residual jadi group_1 (noise-like),
    group_2 (trend-like), res (residue akhir CEEMDAN)."""
    imfs = CEEMDAN(trials=trials, parallel=False)(residual_series)
    imf_rows = imfs[:-1]
    res = imfs[-1]

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

    return {'group_1': group_1, 'group_2': group_2, 'res': res}
```

Catatan penting, `parallel=False` diset eksplisit pada `CEEMDAN(trials=trials, parallel=False)`.
`PyEMD.CEEMDAN` secara default (`parallel=True`) diam diam memakai `multiprocessing.Pool`, karena `decompose_residual` dipanggil ulang ribuan kali sepanjang loop produksi (satu kali per hari), pembuatan Pool berulang ini terbukti menghabiskan resource named pipe Windows dan menyebabkan crash (`OSError WinError 1450`) di tengah run yang panjang.
Perbaikan ini juga terbukti membuat runtime lebih cepat, bukan hanya lebih stabil (detail angka di Bagian 2.3).

Grouping berbasis ranking relatif ini juga terbukti robust terhadap perubahan skala data, hasil validasi pada dataset lama (2001-2026) dan dataset EIA (1986-2026, ~1.6 kali lebih panjang, mencakup lebih banyak rezim ekstrem) menghasilkan pola split IMF yang identik (dibahas Bagian 2.4).

### 2.3. Pemilihan Jumlah Trial (trials=50) dan Sifat Stokastik CEEMDAN

CEEMDAN bersifat stokastik, setiap pemanggilan menambahkan realisasi noise acak yang berbeda beda ke sinyal asli sebanyak `trials` kali, lalu dirata ratakan.
Trial lebih banyak umumnya berarti hasil dekomposisi lebih stabil, tapi biayanya juga lebih mahal, waktu komputasi CEEMDAN mendominasi seluruh pipeline (estimasi penuh 62 jam untuk trials=100 pada dataset EIA, single process).

Pengujian awal (v1) sempat memberi kesimpulan keliru, "trials=50 berisiko", karena metrik pembandingnya adalah residue mentah, sementara jumlah IMF yang dihasilkan CEEMDAN sendiri berbeda beda antar run (9 vs 10 vs 11 IMF) bahkan pada trials yang sama, sehingga residue akhir dua run tidak bisa dibandingkan apple to apple.

Pengujian ulang (v2) memperbaiki metrik pembanding, membandingkan hasil GRUP (noise_group dan trend_group via ZCR median split, level yang benar benar dipakai model nanti, bukan IMF/residue mentah), 3 run pada window full series 6400 baris, A dan B sama sama trials=50 (realisasi noise berbeda), C trials=100.
Hasilnya sebagai berikut.

Variasi hasil grup antara trials=50 vs trials=100, sekitar 22.7% (RMSE relatif terhadap std sinyal).
Variasi ALAMI CEEMDAN pada trials yang SAMA (dua run trials=50, realisasi noise acak berbeda), sekitar 21.25%.

Selisih keduanya hanya sekitar 1.5 poin persentase, artinya menaikkan trials dari 50 ke 100 TIDAK memberi perbaikan berarti, karena sifat stokastik CEEMDAN sendiri sudah menyumbang variasi yang jauh lebih besar daripada variasi akibat jumlah trial.
Mengurangi variasi run ke run ini secara signifikan membutuhkan trials jauh lebih tinggi (200 sampai 500+), tidak realistis secara komputasi mengingat estimasi runtime yang sudah 31 jam pada trials=50.

Solusi yang dipakai bukan menaikkan trials, melainkan mengunci reproducibility lewat seed harian tetap (`seed = SEED_BASE + t`, sudah dibahas Bagian 1.2), sehingga pipeline menghasilkan angka yang identik persis setiap kali dijalankan ulang, meski sifat stokastik CEEMDAN antar seed yang berbeda tetap ada dan diterima sebagai karakteristik algoritma, bukan sesuatu yang dihilangkan.

### 2.4. Hasil Validasi Modul (Dataset EIA 1986-2026)

Validasi modul dijalankan sekali secara penuh (trials=100, akurasi diutamakan di atas kecepatan karena ini validasi sekali jalan, bukan loop produksi) pada Residual full series (10.191 baris, bukan expanding window, khusus untuk memastikan modul benar secara matematis).
Ringkasan hasilnya sebagai berikut, lengkap di `evaluations/statistical/ceemdan/ceemdan-decomposition-report.txt`.

CEEMDAN menghasilkan 11 IMF, jumlah yang identik dengan hasil dekomposisi pada dataset lama (2001-2026), walau dataset EIA ini sekitar 1.6 kali lebih panjang dan mencakup rezim pasar jauh lebih volatile.

Skor komposit (rata rata rank ZCR dan rank SampEn) turun monoton dari IMF 1 (skor 11, paling noise) ke IMF 11 (skor 1, paling mirip tren), tidak ada ambiguitas di titik potong median.
Split grouping menghasilkan IMF 1 sampai 5 masuk group_1 (noise-like), IMF 6 sampai 11 masuk group_2 (trend-like), pola split 5 banding 6 yang identik dengan dataset lama.

| IMF | ZCR | Sample Entropy | Skor komposit | Grup |
|---|---:|---:|---:|---|
| 1 | 0.6412 | 0.6659 | 11.0 | group_1 (noise) |
| 2 | 0.3099 | 0.4681 | 9.5 | group_1 (noise) |
| 3 | 0.1570 | 0.4821 | 9.5 | group_1 (noise) |
| 4 | 0.0762 | 0.3655 | 8.0 | group_1 (noise) |
| 5 | 0.0315 | 0.1723 | 7.0 | group_1 (noise) |
| 6 | 0.0143 | 0.0596 | 6.0 | group_2 (trend) |
| 7 | 0.0075 | 0.0372 | 5.0 | group_2 (trend) |
| 8 | 0.0036 | 0.0177 | 4.0 | group_2 (trend) |
| 9 | 0.0019 | 0.0051 | 2.5 | group_2 (trend) |
| 10 | 0.0010 | 0.0076 | 2.5 | group_2 (trend) |
| 11 | 0.0004 | 0.0035 | 1.0 | group_2 (trend) |

Dua pengecekan identitas matematis dilakukan, keduanya lolos dengan error di level floating point noise saja (bukan kesalahan struktural).

Identity check, Residual = group_1 + group_2 + res, error maksimum 1.42e-14.
Sanity check tambahan, Close = Trend + group_1 + group_2 + res, error maksimum 2.84e-14.

Konsistensi hasil (jumlah IMF sama, titik split sama) antara dua dataset dengan panjang dan karakter historis yang jauh berbeda menjadi bukti tambahan bahwa desain grouping berbasis ranking relatif memang robust terhadap perubahan skala dan karakter data, sesuai tujuan desain awalnya.

### 2.5. Keputusan Final

N=2 grup tetap (group_1 noise-like, group_2 trend-like) dipilih demi menjaga dimensi fitur tetap konstan sepanjang expanding window, sekaligus tervalidasi robust lintas dua dataset berbeda panjang dan karakter.
`trials=50` dipertahankan untuk loop produksi, dikonfirmasi empiris bahwa menaikkan ke trials=100 tidak memberi perbaikan berarti dibanding variasi stokastik alami CEEMDAN sendiri.
Reproducibility dijaga lewat seed harian tetap, bukan lewat menghilangkan sifat stokastik CEEMDAN itu sendiri.
`parallel=False` wajib dipakai di seluruh tahap (validasi modul maupun loop produksi), bukan pilihan performa semata, melainkan keharusan untuk menghindari crash pada run jangka panjang.

### 2.6. Visualisasi

Spektrum seluruh IMF hasil dekomposisi, ditumpuk per subplot, warna menandai kelompok (salmon untuk group_1 noise-like, biru untuk group_2 trend-like), (`evaluations/graphical/ceemdan/01_ceemdan_imf_spectrum_grouped.png`).

Seluruh IMF mentah ditumpuk dalam satu plot, garis solid menandai IMF group_1, garis putus putus menandai IMF group_2 (`evaluations/graphical/ceemdan/02_ceemdan_all_imf_overlay.png`).

Hasil akhir setelah dijumlah per grup (group_1, group_2, res) dibandingkan terhadap Residual asli hasil LOWESS (`evaluations/graphical/ceemdan/03_ceemdan_groups_vs_residual.png`).

Perbandingan ZCR dan Sample Entropy per IMF dalam satu grafik dual axis, garis putus putus menandai batas potong antara group_1 dan group_2 (`evaluations/graphical/ceemdan/04_entropy_zcr_comparison.png`).

## 3. Feature Engineering (Windowing/Lag)

### 3.1. Konsep dan Alasan Pemilihan

Setelah tahap LOWESS dan CEEMDAN, tiap baris dataset sudah punya 4 komponen (Trend, group_1, group_2, res) untuk satu titik waktu t.
Model deep learning butuh konteks historis, bukan cuma nilai komponen di satu titik waktu saja, supaya bisa mempelajari pola dari beberapa hari sebelumnya untuk memprediksi hari berikutnya.
Tahap ini mengubah keempat komponen tadi menjadi pasangan input-output (supervised) lewat teknik windowing atau lag, mengambil n hari ke belakang sebagai input (X), dan nilai komponen pada hari itu sendiri sebagai target (y).

```
X = [komponen_(t-n), ..., komponen_(t-1)]   (lag_1 sampai lag_n)
y = komponen_t
```

Windowing dilakukan SEBELUM split Train/Test/Unseen (bukan sesudah), dilakukan di atas satu series kontinu penuh.
Alasannya, lag hanya menoleh ke belakang jadi tidak ada risiko kebocoran data masa depan di urutan manapun ia dilakukan, tapi kalau windowing dibuat setelah split (dikerjakan terpisah per partisi), baris baris awal Test dan Unseen akan kehilangan akses ke lag dari ekor partisi sebelumnya, dan baris baris itu jadi terbuang percuma.

`actual_close` (harga penutupan asli) turut disertakan di tiap baris, tapi HANYA sebagai referensi ground truth untuk evaluasi akhir setelah hasil forecast dari semua komponen dijumlah kembali (rekonstruksi aditif), BUKAN sebagai target training bagi model manapun.

### 3.2. Kode Implementasi (Windowing per Komponen, Skema Thesis)

Sumber, `D:\Coding\#bigdata\crude-oil-forecasting-thesis\12-ceemdan-feature-engineering.py`, dijalankan pada dataset hasil expanding window (Bagian 1 dan 2 gabungan, `WTI-Expanding-LOWESS-CEEMDAN-EIA.csv`).
Salinan lengkap script disimpan di `references/scripts/12-ceemdan-feature-engineering.py`.

Untuk tiap panjang window n (dua kandidat diuji, n=10 dan n=20) dan tiap komponen, dibentuk lag_1 sampai lag_n dan satu kolom target.

```python
COMPONENTS = ['Trend', 'group_1', 'group_2', 'res']
WINDOW_LENGTHS = [10, 20]

def build_windowed_dataset(df, n, components):
    """Untuk tiap target index t (mulai dari index n), bentuk lag_1..lag_n
    dan target_<komponen> untuk SETIAP komponen, plus actual_close
    (referensi evaluasi, bukan target training)."""
    dates = df['Date'].values
    actual_close = df['actual_close'].values
    comp_arrays = {c: df[c].values for c in components}

    rows = []
    for t in range(n, len(df)):
        row = {'Date': dates[t]}
        for c in components:
            lag_window = comp_arrays[c][t - n:t]  # komponen_(t-n) ... komponen_(t-1)
            row.update({f'{c}_lag_{n - i}': lag_window[i] for i in range(n)})
            row[f'target_{c}'] = comp_arrays[c][t]
        row['actual_close'] = actual_close[t]
        rows.append(row)

    return pd.DataFrame(rows)
```

Output skema thesis ini satu file per panjang window, berisi lag dan target keempat komponen sekaligus dalam satu baris (bukan 4 file terpisah per komponen), supaya konsisten dengan format dataset lain di repo thesis.
Untuk n=10, hasilnya `WTI-CEEMDAN-FE-n10.csv`, 9.681 baris x 46 kolom (10 lag x 4 komponen = 40 kolom, plus 4 kolom target per komponen, plus `actual_close`, plus `Date`).
Sanity check identitas juga dilakukan ulang setelah windowing untuk memastikan tidak ada pergeseran index, `target_Trend + target_group_1 + target_group_2 + target_res` dibandingkan `actual_close`, error maksimum 2.84e-14, level floating point noise, lolos.

### 3.3. Adaptasi untuk Proyek Ini (Reshape ke Input Sequence, Bukan Skema Per-Komponen)

Dataset `WTI-CEEMDAN-FE-n10.csv` inilah yang dipakai sebagai satu satunya sumber data proyek final task ini (`dataset/WTI-CEEMDAN-FE-n10.csv`, disalin apa adanya dari output Bagian 3.2 di atas, tidak diproses ulang).
Namun skema pemakaiannya SENGAJA berbeda dari repo thesis.

Repo thesis (skema aslinya), tiap komponen dilatih model TERPISAH (4 komponen, model masing masing), lalu hasil forecast keempatnya dijumlah kembali (rekonstruksi aditif) untuk mendapat prediksi harga akhir.
Proyek final task ini (keputusan desain, replikasi persis skema eksperimen 8 arsitektur pada `experiment-argument-ID.md`), satu model per arsitektur (8 arsitektur total) langsung memprediksi `actual_close` dari keempat komponen sekaligus sebagai input multivariat, BUKAN 8 arsitektur x 4 komponen terpisah.

Baris pre-windowed dari `WTI-CEEMDAN-FE-n10.csv` (yang tadinya berbentuk flat, 40 kolom lag + 4 kolom target per komponen) di-reshape menjadi tensor 3 dimensi `(batch, 10, 4)`, 10 mewakili panjang lookback (timestep), 4 mewakili jumlah komponen (fitur).
Target yang dipakai untuk training langsung `actual_close`, bukan salah satu `target_<komponen>`.

Sumber, `04-fe-and-split.py` (proyek ini), fungsi `build_xy`.

```python
LOOKBACK = 10
COMPONENTS = ["Trend", "group_1", "group_2", "res"]
N_FEATURES = len(COMPONENTS)

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
```

Urutan timestep di tensor ini penting, `t=0` (indeks pertama sumbu waktu) diisi `lag_10` (observasi PALING LAMA, t-9), `t=9` (indeks terakhir) diisi `lag_1` (observasi PALING BARU, t-0, hari tepat sebelum target).
Susunan kronologis maju ini (lama ke baru) mengikuti konvensi umum input sequence model deep learning (RNN, TCN, Transformer, dan seterusnya), supaya model membaca window persis seperti urutan waktu aslinya.

Integritas rekonstruksi aditif tetap dicek ulang di script ini sebelum split dan training dimulai, `target_Trend + target_group_1 + target_group_2 + target_res` dibandingkan `actual_close`, error maksimum di bawah 1e-6 (ambang assert di kode), lolos.

### 3.4. Keputusan Final

Window length n=10 dipilih sebagai satu satunya panjang lookback yang dipakai proyek ini (n=20 tetap dihasilkan skema thesis sebagai kandidat, tapi tidak dipakai lebih lanjut di proyek final task ini), konsisten dengan keputusan `LOOKBACK = 10` pada replikasi skema `experiment-argument-ID.md`.
Windowing dilakukan sebelum split (bukan sesudah) untuk mencegah baris awal tiap partisi kehilangan akses ke lag dari ekor partisi sebelumnya.
Skema pemakaian proyek ini (satu model multivariat memprediksi `actual_close` langsung dari 4 komponen sekaligus) BUKAN skema rekonstruksi aditif per komponen ala thesis, keputusan ini dibuat sadar demi mereplikasi persis desain eksperimen 8 arsitektur yang jadi acuan tugas, bukan karena skema thesis dianggap keliru.

## 4. Split Train, Test, Unseen

### 4.1. Konsep dan Alasan Pemilihan

Data time series tidak boleh dipecah acak (random split) seperti data tabular biasa, karena urutan waktu itu sendiri adalah bagian dari informasi yang harus dijaga, model tidak boleh "mengintip" data masa depan saat training, dan evaluasi harus mensimulasikan kondisi forecasting nyata (memprediksi hari yang belum pernah dilihat, memakai data yang sudah lewat saja).
Karena itu split dilakukan secara KRONOLOGIS, potongan pertama (paling lama) untuk Train, potongan berikutnya untuk Test, potongan terakhir (paling baru) untuk Unseen, tanpa pengacakan urutan sama sekali.

Proyek ini menjalankan DUA skema split terpisah (dua "run tag"), bukan satu.

`full`, memakai SELURUH baris dataset (`WTI-CEEMDAN-FE-n10.csv`, 9.681 baris, 1988-01-11 sampai 2026-06-29), dibagi 80/10/10 menjadi Train, Test, Unseen.
`wu`, memakai SUBSET dataset yang dipotong sampai `Date <= 2020-02-10`, dibagi 80/20 menjadi Train, Test SAJA (tanpa Unseen), meniru rentang data dan pola pembagian data pada studi acuan benchmark Wu et al. (`benchmark-wu-comparison.md`).

Dua skema ini punya tujuan berbeda, `full` untuk mengukur performa model pada rentang waktu paling panjang dan terbaru yang tersedia, `wu` khusus untuk perbandingan apple-to-apple yang lebih adil terhadap studi acuan, karena studi itu memakai rentang data yang berhenti di 2020-02-10.

### 4.2. Kode Implementasi

Sumber, `04-fe-and-split.py` (proyek ini), fungsi `main`.

Split dilakukan dengan slicing index biasa (bukan fungsi split library scikit-learn yang defaultnya mengacak), karena data sudah terurut kronologis berdasarkan `Date` sejak awal.

```python
WU_CUTOFF = "2020-02-10"

if tag == "wu":
    df = df[df["Date"] <= WU_CUTOFF].reset_index(drop=True)

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
```

Untuk run `full`, sisa 20% setelah Train dibagi rata dua, 10% Test dan 10% Unseen (`n_test = int((n - n_train) * 0.5)`).
Untuk run `wu`, sisa 20% setelah Train seluruhnya jadi Test, tidak ada Unseen.

Setelah split, dua langkah preprocessing dilakukan, keduanya WAJIB di-fit HANYA pada data Train untuk mencegah kebocoran statistik dari Test/Unseen ke proses training.

```python
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
```

Winsorizing (pemotongan nilai ekstrem ke kuantil 1% dan 99%) dilakukan LEBIH DULU, baru MinMaxScaler dilatih di atas data yang sudah dipotong, keduanya cuma "melihat" statistik Train (kuantil dan min/max), lalu diterapkan (transform) apa adanya ke Test dan Unseen tanpa dihitung ulang.
Fitur (X) dan target (y) diskalakan dengan scaler TERPISAH, karena keduanya bukan besaran yang identik (X adalah 4 komponen CEEMDAN, y adalah harga aktual).

### 4.3. Hasil Split

Ringkasan rentang tanggal dan jumlah baris tiap partisi, lengkap di `dataset/splits/{full,wu}/split-report.md`.

Run `full` (total 9.681 baris).

| Partisi | Rentang tanggal | Jumlah baris |
|---|---|---:|
| Train | 1988-01-11 s.d. 2018-09-24 | 7.744 |
| Test | 2018-09-25 s.d. 2022-08-08 | 968 |
| Unseen | 2022-08-09 s.d. 2026-06-29 | 969 |

Run `wu` (total 8.086 baris, subset `Date <= 2020-02-10`).

| Partisi | Rentang tanggal | Jumlah baris |
|---|---|---:|
| Train | 1988-01-11 s.d. 2013-08-28 | 6.468 |
| Test | 2013-08-29 s.d. 2020-02-10 | 1.618 |

Catatan penting soal jumlah baris run `wu`, Wu et al. memakai 8.596 sampel pada rentang 1986-01-02 sampai 2020-02-10 di studi acuan, sedangkan proyek ini menghasilkan 8.086 baris pada rentang tanggal yang sama.
Selisih 510 baris ini BUKAN kesalahan, melainkan konsekuensi dari warmup expanding window (Bagian 1.2 dan 3.1, 500 baris pertama dilewati saat LOWESS/CEEMDAN, ditambah 10 baris lagi hilang saat windowing/lag Bagian 3), harus disebutkan secara eksplisit di setiap perbandingan dengan Wu et al. supaya jujur secara metodologis.

Kuantil winsorizing (1% dan 99%) yang dihasilkan dari data Train tiap run sebagai berikut.

| Fitur | Run full, clip bawah | Run full, clip atas | Run wu, clip bawah | Run wu, clip atas |
|---|---:|---:|---:|---:|
| Trend | 12.9102 | 123.2291 | 12.4765 | 127.8532 |
| group_1 | -11.8475 | 11.2121 | -12.7207 | 10.0250 |
| group_2 | -7.8074 | 6.0424 | -7.2446 | 6.2208 |
| res | -0.5346 | 1.1139 | -0.1880 | 1.1816 |

### 4.4. Keputusan Final

Dua run terpisah (`full` dan `wu`) dipertahankan sebagai desain final, bukan cuma satu run, karena keduanya menjawab pertanyaan yang berbeda, `full` mengukur performa model senyata mungkin (rentang data terpanjang dan terbaru), `wu` menyediakan basis perbandingan yang adil terhadap studi acuan Wu et al. meski tetap tidak identik (lihat catatan selisih baris di atas, serta catatan look-ahead leakage Wu et al. di `benchmark-wu-comparison.md`).
Split kronologis (bukan acak) dan preprocessing fit-hanya-di-Train dipertahankan tanpa pengecualian di kedua run, sebagai prinsip anti-leakage yang konsisten sepanjang pipeline.

## 5. Training 8 Arsitektur Deep Learning

### 5.1. Konsep dan Alasan Pemilihan

Proyek ini melatih dan membandingkan 8 arsitektur deep learning sekaligus (bukan satu model saja), mereplikasi persis desain eksperimen acuan tugas.
Kedelapan arsitektur mewakili beberapa keluarga pendekatan yang berbeda cara "membaca" window 10 hari (Bagian 3 dan 4).

MLP (Multi-Layer Perceptron), baseline paling sederhana, HANYA membaca timestep terbaru (`lag_1`, t-0) dari tiap window, bukan seluruh 10 hari, karena arsitekturnya memang bukan untuk data sequence.
RNN, LSTM, BiLSTM, GRU, keluarga recurrent, membaca window secara berurutan hari demi hari, mempertahankan "memori" dari hari-hari sebelumnya lewat hidden state.
TCN (Temporal Convolutional Network), memakai convolution kausal (hanya melihat masa lalu, bukan masa depan) dengan dilasi bertingkat, menangkap pola lokal dan jangka menengah sekaligus.
Transformer dan Informer, keluarga attention, membaca seluruh window sekaligus secara paralel (bukan berurutan), memakai mekanisme attention untuk menimbang pentingnya tiap hari terhadap prediksi.

Kedelapan model memprediksi target yang SAMA, `actual_close`, dari input yang SAMA (4 komponen CEEMDAN, window 10 hari, Bagian 3 dan 4), sehingga hasil metrik antar arsitektur bisa dibandingkan secara adil (apple to apple), bukan model dilatih dengan data atau target berbeda beda.

### 5.2. Kode Implementasi (Arsitektur dan Prosedur Training)

Sumber, `05-dl-model-training.py` (proyek ini), adaptasi dari `crude-oil-forecasting-DL/05-dl-model-training.py` (7 arsitektur) dan `08-rnn-model-training.py` (kelas RNN), kelas model disalin verbatim, hanya `N_FEATURES` diganti dari 7 fitur makro menjadi 4 komponen CEEMDAN.

Konfigurasi utama (sama untuk semua arsitektur dan kedua run).

```python
SEED = 42
LOOKBACK = 10          # T, panjang window
N_FEATURES = 4          # jumlah komponen CEEMDAN
BATCH_SIZE = 32
MAX_EPOCHS = 200        # default, bisa dioverride --max-epochs
PATIENCE = 20           # early stopping patience
LR = 1e-3               # Adam initial learning rate
LR_FACTOR = 0.5         # ReduceLROnPlateau reduction factor
LR_PATIENCE = 10        # epoch tanpa perbaikan sebelum LR diturunkan
LR_MIN = 1e-5
VAL_FRACTION = 0.10     # 10% EKOR Train dipakai validasi
```

Validasi diambil dari EKOR data Train (10% baris terakhir secara kronologis), BUKAN diacak dari seluruh Train, supaya validasi tetap mensimulasikan forecasting maju ke depan, konsisten dengan prinsip anti-leakage split kronologis di Bagian 4.

```python
n_val = int(len(X_train_full) * VAL_FRACTION)
n_tr = len(X_train_full) - n_val

X_tr_seq, y_tr_seq = X_train_full[:n_tr],  y_train_full[:n_tr]
X_val_seq, y_val_seq = X_train_full[n_tr:], y_train_full[n_tr:]

# MLP hanya pakai timestep TERBARU (t-0, lag_1) tiap window
X_tr_flat = X_tr_seq[:, -1, :]
X_val_flat = X_val_seq[:, -1, :]
```

Loop training memakai loss MSE, optimizer Adam, gradient clipping (`max_norm=1.0`, mencegah gradient meledak terutama pada arsitektur recurrent), scheduler `ReduceLROnPlateau` (menurunkan learning rate kalau validasi loss stagnan), dan early stopping (menghentikan training kalau validasi loss tidak membaik selama `PATIENCE=20` epoch berturut turut, lalu mengembalikan bobot model ke titik validasi loss terbaik, bukan bobot di epoch terakhir).

```python
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
      self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
      self.counter += 1
    return self.counter >= self.patience

  def restore_best(self, model):
    if self.best_state:
      model.load_state_dict(self.best_state)
```

Seed (`torch.manual_seed(42)`, `np.random.seed(42)`) dikunci di awal setiap run untuk reproducibility, GPU CUDA dipakai otomatis kalau tersedia (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).

### 5.3. Metrik Evaluasi

Tujuh metrik dihitung di skala harga asli USD (setelah inverse transform dari scaler Bagian 4), bukan di skala 0-1 hasil MinMaxScaler.

MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error), rata rata kesalahan absolut dalam USD.
MAPE (Mean Absolute Percentage Error) dan SMAPE (Symmetric MAPE), kesalahan dalam persentase, SMAPE lebih stabil saat harga mendekati nol.
DA (Directional Accuracy), persentase hari yang arah pergerakannya (naik/turun dibanding hari sebelumnya) ditebak benar, dihitung dari `Trend` komponen `lag_1` sebagai referensi harga hari sebelumnya (bukan `actual_close` hari sebelumnya, karena `actual_close` hari sebelumnya tidak ikut disimpan sebagai kolom terpisah, referensi ini didekati dari komponen Trend yang paling dekat dengan level harga).
MASE (Mean Absolute Scaled Error), MAE dibagi MAE model naif (tebak harga besok sama dengan harga hari ini), MASE di bawah 1 berarti model mengalahkan tebakan naif.
R2 (koefisien determinasi), seberapa besar variansi target berhasil dijelaskan model.

Uji statistik Diebold-Mariano (koreksi sampel kecil Harvey et al. 1997) dijalankan berpasangan antar seluruh 28 kombinasi model (8 pilih 2), untuk menguji apakah selisih kesalahan kuadrat (squared error) antar dua model signifikan secara statistik, bukan cuma kebetulan variasi run.

Permutation feature importance juga dihitung per model (10 kali ulang per fitur, `N_PERM_REPEATS=10`), mengacak satu fitur di seluruh sampel lalu mengukur seberapa besar RMSE memburuk, fitur yang penting akan membuat RMSE naik signifikan saat diacak.

### 5.4. Hasil Training

Ringkasan performa akhir tiap arsitektur di skala harga asli USD, lengkap di `evaluations/statistical/model-train/{full,wu}/01_metrics_summary.md`.

Run `full`, Test set (968 baris, 2018-09-25 s.d. 2022-08-08).

| Model | MAE | MAPE% | DA% | R2 |
|---|---:|---:|---:|---:|
| MLP | 3.1983 | 6.9060 | 80.48 | 0.9440 |
| RNN | 2.7962 | 5.5286 | 86.78 | 0.9567 |
| LSTM | 3.4650 | 6.8126 | 84.30 | 0.9157 |
| BiLSTM | 3.0547 | 6.0119 | 85.95 | 0.9404 |
| GRU | 3.1805 | 6.3740 | 84.71 | 0.9387 |
| TCN | 3.1283 | 6.5207 | 84.40 | 0.9375 |
| Transformer | 3.5398 | 6.4038 | 82.33 | 0.9257 |
| Informer | 3.4012 | 6.7074 | 81.51 | 0.9343 |

Run `full`, Unseen set (969 baris, 2022-08-09 s.d. 2026-06-29).

| Model | MAE | MAPE% | DA% | R2 |
|---|---:|---:|---:|---:|
| MLP | 3.1099 | 3.7210 | 85.04 | 0.6527 |
| RNN | 2.9908 | 3.6025 | 87.00 | 0.6886 |
| LSTM | 3.2860 | 3.9014 | 86.38 | 0.5805 |
| BiLSTM | 3.0320 | 3.6093 | 88.13 | 0.6535 |
| GRU | 3.0896 | 3.6906 | 86.38 | 0.6455 |
| TCN | 3.0933 | 3.7214 | 88.44 | 0.6538 |
| Transformer | 3.3258 | 4.0147 | 83.08 | 0.6461 |
| Informer | 3.5364 | 4.2453 | 80.39 | 0.5749 |

Run `wu`, Test set (1.618 baris, 2013-08-29 s.d. 2020-02-10).

| Model | MAE | MAPE% | DA% | R2 |
|---|---:|---:|---:|---:|
| MLP | 2.4679 | 4.3933 | 81.89 | 0.9722 |
| RNN | 1.2945 | 2.3309 | 88.57 | 0.9908 |
| LSTM | 1.5556 | 2.7669 | 86.40 | 0.9889 |
| BiLSTM | 1.3788 | 2.4473 | 87.70 | 0.9902 |
| GRU | 1.4934 | 2.6634 | 86.46 | 0.9891 |
| TCN | 1.2462 | 2.2432 | 89.18 | 0.9910 |
| Transformer | 1.7047 | 2.9898 | 85.78 | 0.9868 |
| Informer | 1.6921 | 2.9908 | 84.67 | 0.9870 |

RNN terbaik di run `full` Test (MAPE 5.53%, DA 86.78%, R2 0.9567), TCN terbaik di run `wu` Test (MAPE 2.24%, DA 89.18%, R2 0.9910).
MLP (baseline paling sederhana) konsisten menjadi model terlemah di kedua run, sesuai dugaan awal karena MLP tidak melihat konteks historis sama sekali (hanya `lag_1`), sementara model lain memanfaatkan seluruh window 10 hari.

Catatan penting soal R2 yang anjlok drastis di Unseen (0.55 sampai 0.69) dibanding Test (0.92 sampai 0.96) pada run `full`, meski MAPE justru MEMBAIK di Unseen (3.6 sampai 4.2% berbanding 5.5 sampai 6.9% di Test).
Ini BUKAN model memburuk, melainkan variansi target Unseen jauh lebih kecil dari Test (Test memuat crash harga COVID 2020, rentang harga jauh lebih liar), R2 sensitif terhadap variansi target sementara MAPE tidak, jadi dua metrik ini harus dibaca berdampingan, bukan R2 saja.

Directional Accuracy (80-89% di kedua run) jauh lebih tinggi dibanding eksperimen referensi 7 fitur makro (46-49% di studi itu), diduga kuat karena komponen Trend (hasil LOWESS) pada `lag_1` sudah sangat dekat dengan `actual_close` hari berikutnya, sinyal arah pergerakan kemungkinan "bocor" dari proses dekomposisi itu sendiri, dibahas lebih lanjut di tahap XAI (bagian berikutnya).

### 5.5. Visualisasi

Kurva loss Train dan Validation sepanjang epoch, seluruh 8 model dalam satu figure (`evaluations/graphical/model-train/{full,wu}/training_curves.png`).

Actual vs Predicted, seluruh 8 model dalam satu figure per split (`evaluations/graphical/model-train/{full,wu}/actual_vs_predicted_{test,unseen}.png`), serta versi tunggal per model di folder `actual-vs-predicted-{test,unseen}-set/`.

Perbandingan 7 metrik (MAE, MAPE, SMAPE, RMSE, DA, MASE, R2) antar 8 model dalam satu figure (`evaluations/graphical/model-train/{full,wu}/metrics_comparison_{test,unseen}.png`).

Heatmap hasil uji Diebold-Mariano 8x8 model (`evaluations/graphical/model-train/{full,wu}/dm_heatmap_{test,unseen}.png`), warna merah/biru menandai model mana yang signifikan lebih unggul di tiap pasangan.

Permutation feature importance, seluruh 8 model dalam satu figure per split (`evaluations/graphical/model-train/{full,wu}/feature_importance_all_models_{test,unseen}.png`), serta versi tunggal per model di folder `feature-importance-{test,unseen}-set/`.

### 5.6. Keputusan Final

Delapan arsitektur dipertahankan sebagai cakupan final (bukan dipangkas ke beberapa model saja), untuk mereplikasi persis skema eksperimen acuan tugas dan memberi basis perbandingan paling lengkap sebelum tahap XAI.
Skema training identik untuk seluruh arsitektur di kedua run (hyperparameter loop training sama, hanya `MODEL_NAMES` dan kelas model yang berbeda), memastikan perbandingan performa antar arsitektur adil, bukan salah satu diuntungkan setup training yang berbeda.
Anomali R2 anjlok di Unseen (run `full`) diterima sebagai karakteristik data (variansi target lebih kecil), dijelaskan secara eksplisit di laporan, bukan disembunyikan atau dianggap kegagalan model.

## 6. XAI (Explainable AI, SHAP dan Integrated Gradients)

### 6.1. Konsep dan Alasan Pemilihan

Bagian 5 sudah mengukur SEBERAPA AKURAT tiap arsitektur, tapi belum menjawab MENGAPA satu arsitektur lebih akurat dari yang lain, model deep learning pada dasarnya kotak hitam, prediksinya tidak otomatis bisa dijelaskan hanya dari metrik akurasi.
Tahap XAI ini membedah 8 model terlatih dari Bagian 5 untuk melihat bagian input mana (komponen CEEMDAN mana, dan hari ke berapa dalam window 10 hari) yang paling berpengaruh terhadap tiap prediksi, lalu mengaitkan pola atribusi ini dengan performa (MAE) tiap model.

Dua metode atribusi dipakai sekaligus, bukan cuma satu, supaya temuan tidak bergantung pada bias satu metode saja.

SHAP (SHapley Additive exPlanations, lewat `shap.GradientExplainer`), berbasis teori permainan (game theory), mengukur kontribusi tiap fitur dengan membandingkan prediksi terhadap sekumpulan data latar belakang (background).
Integrated Gradients (lewat `captum`), berbasis integral gradien sepanjang jalur dari titik baseline (nol) ke input aktual, metode aksiomatik yang populer untuk model neural network.

Atribusi mentah (satu angka per timestep per komponen per sampel) lalu diringkas ke dua sudut pandang berbeda.

Atribusi PER KOMPONEN (dijumlah antar 10 timestep), menjawab, komponen CEEMDAN mana (Trend, group_1, group_2, res) yang paling penting bagi tiap model.
Atribusi PER TIMESTEP (dijumlah antar 4 komponen), menjawab, hari ke berapa dalam window 10 hari yang paling penting, khusus untuk 7 model sequence (MLP tidak punya dimensi timestep karena hanya membaca `lag_1`).

Skor konsentrasi (1 dikurangi entropi Shannon ternormalisasi, skala 0 sampai 1) dihitung dari kedua sudut pandang ini, 0 berarti atribusi menyebar rata ke semua komponen/timestep, 1 berarti atribusi terpusat penuh ke satu komponen/timestep saja.
Skor konsentrasi ini lalu dikorelasikan (Spearman) terhadap MAE tiap model, untuk menguji apakah model yang atribusinya lebih terpusat cenderung lebih akurat.

### 6.2. Kode Implementasi

Sumber, `09-xai-explainability.py` (proyek ini), adaptasi dari `crude-oil-forecasting-DL/09-xai-explainability.py`, kelas model disalin verbatim dari Bagian 5, dataset dimuat langsung dari `dataset/splits/{tag}/splits.npz` (sudah pre-windowed, tidak perlu membangun ulang sequence).

```python
XAI_EVAL_SAMPLE_SIZE = 200   # subsample window per split, SHAP+IG mahal
SHAP_BACKGROUND_SIZE = 200
SHAP_NSAMPLES = 50
IG_STEPS = 50

FEATURE_NAMES = ["Trend", "IMF_Group1", "IMF_Group2", "Residual"]

def compute_shap_attributions(model, X_bg, X_eval, device, n_repeats, seed=SEED):
  bg_t = torch.tensor(X_bg, dtype=torch.float32, device=device)
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
```

Informer punya lapisan `ProbSparseAttention` yang memakai `torch.randint` saat memilih kandidat key, jadi forward pass-nya TETAP stokastik meski model sudah dalam mode eval (beda dengan 7 model lain yang deterministik penuh di eval mode).
Untuk menstabilkan atribusinya, Informer dirata rata dari 10 kali pengulangan (`STABILITY_REPEATS`), model lain cukup 1 kali karena hasilnya sudah pasti sama.

```python
STABILITY_REPEATS = {
    "MLP": 1, "RNN": 1, "LSTM": 1, "BiLSTM": 1, "GRU": 1, "TCN": 1,
    "Transformer": 1, "Informer": 10,
}
```

Catatan teknis tambahan, `torch.backends.cudnn.enabled = False` diset khusus saat menghitung atribusi (bukan saat training Bagian 5), karena kernel cuDNN untuk backward RNN/LSTM/GRU menolak berjalan kecuali forward pass juga dalam mode training (yang akan mengaktifkan kembali dropout dan membuat atribusi ikut stokastik), menonaktifkan cuDNN membuat backward pass tetap bisa jalan di mode eval dengan implementasi generik, atribusi jadi deterministik, biayanya bisa diabaikan pada skala model/data proyek ini.

Fungsi skor konsentrasi.

```python
def concentration_score(v):
  """1 - normalized Shannon entropy. 0 = menyebar rata, 1 = terpusat penuh."""
  v = np.asarray(v, dtype=np.float64).reshape(-1)
  if v.sum() <= 0 or len(v) <= 1:
    return 0.0
  p = v / v.sum()
  h = scipy_entropy(p, base=2)
  h_max = math.log2(len(v))
  return float(1.0 - (h / h_max if h_max > 0 else 0.0))
```

### 6.3. Hasil, Atribusi per Komponen (Trend Mendominasi)

Komponen Trend (hasil LOWESS) mendominasi atribusi SHAP di SEMUA 8 model dan SEMUA split yang diuji, lengkap di `evaluations/xai/statistical/{full,wu}/01_feature_attribution_{test,unseen}.csv`.

| Split | Rentang share atribusi Trend (8 model) |
|---|---|
| Run `full`, Test | 60.8% (Transformer) sampai 70.1% (MLP) |
| Run `full`, Unseen | 74.2% (Transformer) sampai 79.1% (MLP) |
| Run `wu`, Test | 74.6% (Transformer) sampai 82.0% (MLP) |

Atribusi Integrated Gradients menunjukkan pola serupa atau bahkan lebih tinggi lagi untuk Trend.
Temuan ini KONSISTEN dengan analisis awal soal Directional Accuracy yang tinggi (Bagian 5.4), komponen Trend pada `lag_1` memang sangat dekat dengan `actual_close` hari berikutnya, dan atribusi SHAP/IG mengonfirmasi model memang benar benar mengandalkan komponen ini jauh di atas 3 komponen lainnya.

### 6.4. Hasil, Konsentrasi Atribusi vs Error (Korelasi Spearman)

Dua sudut pandang konsentrasi diuji terpisah terhadap MAE, hasilnya SANGAT BERBEDA, lengkap di `evaluations/xai/statistical/{full,wu}/04_concentration_vs_error_correlation.txt`.

Konsentrasi ANTAR KOMPONEN (feature concentration, 8 model), TIDAK signifikan berkorelasi dengan MAE di ketiga split.

| Split | Spearman rho | p-value |
|---|---:|---:|
| Run `full`, Test | -0.1429 | 0.7358 |
| Run `full`, Unseen | +0.1190 | 0.7789 |
| Run `wu`, Test | -0.0476 | 0.9108 |

Konsentrasi ANTAR TIMESTEP (timestep concentration, 7 model sequence, tidak termasuk MLP), KONSISTEN signifikan berkorelasi NEGATIF dengan MAE di ketiga split.

| Split | Spearman rho | p-value |
|---|---:|---:|
| Run `full`, Test | -0.7857 | 0.0362 |
| Run `full`, Unseen | -0.8214 | 0.0234 |
| Run `wu`, Test | -0.8571 | 0.0137 |

Rho negatif dan signifikan berarti, semakin tajam model memusatkan atribusinya ke satu timestep tertentu (khususnya `lag_1`, hari paling baru), semakin RENDAH (semakin baik) MAE model itu.
Contoh konkret di run `full` Test, TCN (skor konsentrasi timestep 0.815, MAE 3.177) dan RNN (0.404, MAE 3.029) berada di ujung tinggi konsentrasi dan berkinerja baik, sementara Transformer (0.083, MAE 3.830) dan Informer (0.078, MAE 3.623) berada di ujung rendah konsentrasi (atribusinya menyebar cukup rata sepanjang 10 hari lookback) dan berkinerja lebih buruk di antara model sequence.

### 6.5. Interpretasi Temuan Utama

Hipotesis awal (dipinjam dari studi referensi 7 fitur), model attention kalah karena atribusinya lebih menyebar ANTAR FITUR dibanding model lain, TIDAK terbukti di setup CEEMDAN 4-komponen ini (Bagian 6.4, feature concentration tidak signifikan).
Konsentrasi antar fitur relatif SERAGAM di semua arsitektur, karena Trend selalu mendominasi terlepas dari arsitekturnya (Bagian 6.3), bukan pembeda yang bisa menjelaskan perbedaan performa antar model.

Temuan yang justru KONSISTEN dan SIGNIFIKAN adalah konsentrasi ANTAR TIMESTEP, bukan konsentrasi antar fitur, yang membedakan leaderboard performa.
Model yang secara arsitektural memaksa bobot lebih besar ke observasi paling baru cenderung menang, TCN lewat lapisan dilated convolution terakhir yang secara struktural paling dekat ke `lag_1`, RNN lewat sifat vanishing gradient alaminya yang justru menguntungkan di sini karena membuat pengaruh hari-hari lama meluruh secara natural.
Model attention (Transformer, Informer) yang secara arsitektural dirancang menyebar perhatian ke SELURUH window justru kalah, karena tidak secara eksplisit mengutamakan recency, padahal sinyal paling informatif (Trend) justru paling kuat di `lag_1`.

Kesimpulannya, bukan seberapa terpusat model membaca ANTAR KOMPONEN yang menentukan performa, melainkan seberapa terpusat model membaca ANTAR WAKTU (recency bias). Temuan inilah yang mendasari eksperimen mitigasi di Bagian berikutnya (perbaikan recency bias pada Transformer dan Informer).

### 6.6. Visualisasi

Perbandingan share atribusi antar 4 komponen, seluruh 8 model dalam satu figure per split (`evaluations/xai/graphical/{full,wu}/comparison/feature_importance_all_models_{test,unseen}.png`).

Scatter plot konsentrasi atribusi (antar komponen) vs MAE, seluruh 8 model per split (`evaluations/xai/graphical/{full,wu}/comparison/concentration_vs_mae_{test,unseen}.png`).

Atribusi per komponen, SHAP vs Integrated Gradients berdampingan, per model per split (`evaluations/xai/graphical/{full,wu}/feature-attribution/{test,unseen}/`).

Atribusi per timestep (7 model sequence), SHAP vs Integrated Gradients berdampingan, per model per split (`evaluations/xai/graphical/{full,wu}/timestep-attribution/{test,unseen}/`).

Heatmap atribusi SHAP (10 timestep x 4 komponen), per model per split (`evaluations/xai/graphical/{full,wu}/attribution-heatmaps/{test,unseen}/`).

### 6.7. Keputusan Final

Dua metode atribusi (SHAP dan Integrated Gradients) dipertahankan sekaligus, bukan satu saja, karena keduanya konsisten menunjukkan pola yang sama (dominasi Trend, dan konsentrasi timestep yang berkorelasi dengan MAE), memperkuat keyakinan temuan bukan artefak satu metode.
Hipotesis awal soal konsentrasi antar fitur DITINGGALKAN secara eksplisit karena tidak didukung data (Bagian 6.4), diganti dengan hipotesis konsentrasi antar timestep (recency bias) yang didukung data signifikan di ketiga split, keputusan ini jujur mengikuti bukti, bukan mempertahankan hipotesis awal yang ternyata keliru untuk setup proyek ini.

## 7. Benchmark dengan Wu et al. (ICEEMDAN SCA-RVFL)

### 7.1. Konsep dan Alasan Pemilihan

Tugas ini secara eksplisit meminta perbandingan dengan studi acuan, Wu et al. (Improved CEEMDAN SCA-RVFL), yang memakai data harga WTI 2 Januari 1986 sampai 10 Februari 2020 (8.596 sampel harian), split 80/20 Train/Test, metode terbaiknya kombinasi dekomposisi ICEEMDAN dengan ensemble SCA-RVFL (Sine Cosine Algorithm dioptimasi Random Vector Functional Link), dilaporkan pada Table 4 mereka untuk horizon prediksi 1 hari.

Run `wu` (Bagian 4) dirancang khusus untuk perbandingan ini, cutoff `Date <= 2020-02-10` meniru rentang data Wu et al. semirip mungkin, meski tidak identik persis (dibahas Bagian 7.4).
Delapan arsitektur (Bagian 5) dan hasil tuning MLP (dibahas eksperimen lanjutan, motivasi dari Bagian 6.4) pada run ini dipakai sebagai basis perbandingan.

### 7.2. Tabel Perbandingan, Horizon 1

Angka MAPE dan Directional Accuracy kami dikonversi ke skala fraksi (MAPE%/100, DA%/100) supaya sebanding langsung dengan satuan Wu et al. (MAPE fraksi, Dstat fraksi), lengkap di `benchmark-wu-comparison.md`.

| Sumber | Model | MAPE | RMSE | Dstat |
|---|---|---:|---:|---:|
| Wu et al. Table 4, ICEEMDAN | SCA-RVFL (terbaik) | 0.0035 | 0.2801 | 0.9273 |
| Wu et al. Table 4, ICEEMDAN | RVFL | 0.0040 | 0.3187 | 0.9186 |
| Wu et al. Table 4, ICEEMDAN | LSSVR | 0.0047 | 0.3559 | 0.9093 |
| Wu et al. Table 4, ICEEMDAN | BPNN | 0.0045 | 0.3601 | 0.8988 |
| Wu et al. Table 4, ICEEMDAN | ARIMA | 0.0121 | 0.8205 | 0.7720 |
| Wu et al. Table 4, EEMD | SCA-RVFL | 0.0086 | 0.6340 | 0.8045 |
| Wu et al. teks (model tunggal, tanpa decomposition ensemble) | SCA-RVFL | 0.0157 | 1.2183 | 0.7522 |
| Kami, run `wu` | MLP | 0.0439 | 3.2932 | 0.8189 |
| Kami, run `wu` | RNN | 0.0233 | 1.8954 | 0.8857 |
| Kami, run `wu` | LSTM | 0.0277 | 2.0853 | 0.8640 |
| Kami, run `wu` | BiLSTM | 0.0245 | 1.9594 | 0.8770 |
| Kami, run `wu` | GRU | 0.0266 | 2.0638 | 0.8646 |
| Kami, run `wu` | TCN (terbaik) | 0.0224 | 1.8762 | 0.8918 |
| Kami, run `wu` | Transformer | 0.0299 | 2.2688 | 0.8578 |
| Kami, run `wu` | Informer | 0.0299 | 2.2532 | 0.8467 |
| Kami, run `wu` | MLP Tuned | 0.0230 | 1.9330 | 0.8881 |

RMSE kami dan Wu et al. sama sama dalam satuan harga USD per barel, sehingga secara nominal dapat dibandingkan langsung meski dasar perhitungannya berbeda (dibahas Bagian 7.4).

### 7.3. Model Terbaik Kami vs Wu et al.

TCN (MAPE 0,0224, Dstat 0,8918) adalah model terbaik kami pada run `wu`, masih jauh di bawah metode ensemble terbaik Wu et al. (ICEEMDAN SCA-RVFL, MAPE 0,0035, Dstat 0,9273) secara nilai mutlak.
MLP Tuned (MAPE 0,0230, Dstat 0,8881, hasil eksperimen lanjutan) hampir menyamai TCN, mengonfirmasi temuan Bagian 6.4-6.5, arsitektur sederhana pun bisa kompetitif kalau konfigurasi pelatihannya tepat, konsisten dengan temuan XAI bahwa sinyal utama (Trend pada `lag_1`) sudah sangat informatif tanpa memerlukan arsitektur sequential kompleks.

### 7.4. Catatan Kejujuran Perbandingan

Perbandingan di atas BUKAN perbandingan apple to apple, empat faktor struktural membuat angka Wu et al. secara sistematis lebih optimis dari setup kami.

Jumlah baris data berbeda, Wu et al. 8.596 sampel mulai 1986-01-02, dataset kami 8.086 baris pada rentang tanggal yang sama, selisih 510 baris ini akibat warmup expanding window pipeline LOWESS/CEEMDAN kami (Bagian 1.2, 2.2, 4.3), 500 baris pertama dipakai warmup dan tidak menghasilkan dekomposisi valid, ditambah 10 baris lagi hilang saat windowing/lag (Bagian 3.2).

Faktor paling signifikan, dekomposisi Wu et al. mengandung LOOK AHEAD LEAKAGE.
Dekomposisi ICEEMDAN mereka dilakukan SEKALI pada seluruh series data, termasuk bagian yang jadi Test set, komponen hasil dekomposisi pada Test set mereka sudah "melihat" informasi masa depan relatif terhadap titik waktu prediksi, sesuatu yang tidak mungkin terjadi pada skenario forecasting nyata.
Dekomposisi kami sebaliknya memakai EXPANDING WINDOW (Bagian 1.2, 2.2), setiap titik waktu hanya didekomposisi memakai data sampai titik waktu itu sendiri, tanpa informasi masa depan sama sekali.
Faktor ini membuat metrik Wu et al. secara struktural lebih baik dari metrik yang bisa dicapai model manapun pada skenario forecasting yang realistis.

Fitur input berbeda, kami memakai 4 komponen CEEMDAN (Trend, group_1, group_2, res) sebagai input langsung ke satu model deep learning per arsitektur, target `actual_close` langsung (Bagian 3.3).
Wu et al. memakai IMF hasil ICEEMDAN penuh (bukan dikelompokkan jadi 4 komponen) dengan model RVFL TERPISAH per IMF, hasil dijumlahkan sebagai prediksi akhir (skema decomposition ensemble).
Perbedaan skema ini membuat kompleksitas model dan cara memanfaatkan informasi dekomposisi tidak identik.

Cutoff tanggal Test set kami sengaja dipertahankan di 2020-02-10 (bukan memaksa menyamakan jumlah baris 8.596 pertama, keputusan Bagian 4.1), karena kalau dipaksakan menyamakan jumlah baris, Test set akan mencakup 2020-04-20 dengan `actual_close` sebesar -36,98 (harga negatif kontrak berjangka WTI akibat krisis COVID-19), membuat MAPE tidak bermakna secara matematis untuk seluruh model.

### 7.5. Interpretasi

Perbandingan yang lebih adil adalah terhadap baris "model tunggal" Wu et al. TANPA skema decomposition ensemble penuh (SCA-RVFL saja, MAPE 0,0157, Dstat 0,7522), di sini model terbaik kami (TCN) justru UNGGUL pada Dstat (0,8918 vs 0,7522) meski masih kalah pada MAPE.
Ini mengindikasikan sebagian besar keunggulan Wu et al. berasal dari skema decomposition ensemble per IMF plus look ahead leakage, bukan semata dari kapasitas model prediktif itu sendiri.

Kesimpulannya, gap performa terhadap Wu et al. sebagian besar bersifat STRUKTURAL (leakage dekomposisi dan skema ensemble per IMF), bukan murni indikasi model kami secara arsitektural lebih lemah.
Klaim "kalah" atau "menang" secara mentah terhadap Wu et al. tidak tepat tanpa mempertimbangkan keempat faktor Bagian 7.4 di atas.

### 7.6. Keputusan Final

Perbandingan terhadap Wu et al. tetap DITAMPILKAN apa adanya (termasuk angka yang jauh lebih baik dari kami), bukan disembunyikan atau dipoles, karena kejujuran metodologis lebih penting daripada tampil unggul secara angka mentah.
Catatan Bagian 7.4 (empat faktor struktural) WAJIB disertakan setiap kali angka ini dikutip atau dipresentasikan, supaya pembaca tidak salah menyimpulkan proyek ini "kalah" dari Wu et al. tanpa konteks perbedaan metodologi yang mendasar (leakage, skema ensemble, jumlah baris, cutoff tanggal).
