# Perbandingan Benchmark dengan Wu et al. (ICEEMDAN SCA RVFL)

## 1. Setup Penelitian Wu et al.

Penelitian acuan (Wu, Improved CEEMDAN SCA RVFL) memakai data harga minyak mentah WTI dari 2 Januari 1986 sampai 10 Februari 2020, total 8.596 sampel harian.
Split data 80 persen train dan 20 persen test.
Metrik evaluasi yang dilaporkan adalah MAPE (Mean Absolute Percentage Error dalam bentuk fraksi, bukan persen), RMSE (Root Mean Squared Error), dan Dstat (directional statistic, setara DA/directional accuracy dalam bentuk fraksi).

Metode terbaik pada penelitian tersebut adalah kombinasi dekomposisi ICEEMDAN dengan model ensemble SCA RVFL (Sine Cosine Algorithm dioptimasi Random Vector Functional Link), dilaporkan pada Table 4 untuk horizon prediksi 1 hari.

## 2. Setup Kami untuk Run Wu

Untuk membandingkan pada rentang data yang sedekat mungkin, run `wu` pada proyek ini memakai subset data dengan cutoff `Date <= 2020-02-10`, menghasilkan 8.086 baris (bukan 8.596 seperti Wu et al., lihat Bagian 4 untuk penjelasan selisih).
Split 80/20 kronologis, train 6.468 baris (1988-01-11 sampai 2013-08-28) dan test 1.618 baris (2013-08-29 sampai 2020-02-10).

Delapan arsitektur deep learning dilatih pada setup ini, input `(batch, 10, 4)` (lookback 10 hari, 4 komponen CEEMDAN), target `actual_close` langsung, horizon prediksi 1 hari (setara horizon 1 pada Table 4 Wu et al.).

## 3. Tabel Perbandingan, Horizon 1

Angka MAPE dan Dstat kami dikonversi ke skala fraksi (MAPE%/100, DA%/100) agar sebanding langsung dengan satuan Wu et al.

| Sumber | Model | MAPE | RMSE | Dstat |
|---|---|---:|---:|---:|
| Wu et al. Table 4, ICEEMDAN | SCA-RVFL (terbaik) | 0.0035 | 0.2801 | 0.9273 |
| Wu et al. Table 4, ICEEMDAN | RVFL | 0.0040 | 0.3187 | 0.9186 |
| Wu et al. Table 4, ICEEMDAN | LSSVR | 0.0047 | 0.3559 | 0.9093 |
| Wu et al. Table 4, ICEEMDAN | BPNN | 0.0045 | 0.3601 | 0.8988 |
| Wu et al. Table 4, ICEEMDAN | ARIMA | 0.0121 | 0.8205 | 0.7720 |
| Wu et al. Table 4, EEMD | SCA-RVFL | 0.0086 | 0.6340 | 0.8045 |
| Wu et al. teks (model tunggal, tanpa decomposition ensemble) | SCA-RVFL | 0.0157 | 1.2183 | 0.7522 |
| Kami, run wu | MLP | 0.0439 | 3.2932 | 0.8189 |
| Kami, run wu | RNN | 0.0233 | 1.8954 | 0.8857 |
| Kami, run wu | LSTM | 0.0277 | 2.0853 | 0.8640 |
| Kami, run wu | BiLSTM | 0.0245 | 1.9594 | 0.8770 |
| Kami, run wu | GRU | 0.0266 | 2.0638 | 0.8646 |
| Kami, run wu | TCN (terbaik) | 0.0224 | 1.8762 | 0.8918 |
| Kami, run wu | Transformer | 0.0299 | 2.2688 | 0.8578 |
| Kami, run wu | Informer | 0.0299 | 2.2532 | 0.8467 |
| Kami, run wu | MLP Tuned (Bagian 8 laporan utama) | 0.0230 | 1.9330 | 0.8881 |

Catatan skala RMSE, nilai RMSE Wu et al. dan kami sama sama dalam satuan harga USD per barel, sehingga secara nominal dapat dibandingkan langsung meski dasar perhitungannya berbeda (lihat Bagian 4).

## 4. Catatan Kejujuran Perbandingan

Perbandingan di atas bukan perbandingan apple to apple.
Ada empat faktor struktural yang membuat angka Wu et al. secara sistematis lebih optimis dari setup kami.

Pertama, jumlah baris data berbeda.
Wu et al. memakai 8.596 sampel mulai 2 Januari 1986, sementara dataset kami mulai 1988-01-11 (8.086 baris setelah cutoff yang sama).
Selisih ini terjadi karena pipeline dekomposisi CEEMDAN kami memakai skema expanding window, 500 baris pertama data mentah dipakai sebagai warmup dan tidak menghasilkan dekomposisi valid sehingga di-drop dari dataset final.

Kedua, dan ini faktor paling signifikan, dekomposisi Wu et al. mengandung look ahead leakage.
Dekomposisi ICEEMDAN pada penelitian mereka dilakukan sekali pada seluruh series data, termasuk bagian yang menjadi test set.
Ini berarti komponen hasil dekomposisi pada test set sudah "melihat" informasi dari masa depan relatif terhadap titik waktu prediksi, sesuatu yang tidak mungkin terjadi pada skenario forecasting nyata.
Dekomposisi kami sebaliknya memakai expanding window, setiap titik waktu hanya didekomposisi memakai data sampai titik waktu tersebut, tanpa informasi masa depan.
Faktor ini membuat metrik Wu et al. secara struktural lebih baik dari metrik yang bisa dicapai model manapun pada skenario forecasting realistis.

Ketiga, fitur input berbeda.
Kami memakai 4 komponen CEEMDAN (Trend hasil smoothing LOWESS, IMF Group 1, IMF Group 2, Residual) sebagai input langsung ke model deep learning, target `actual_close` langsung.
Wu et al. memakai IMF hasil ICEEMDAN penuh (bukan dikelompokkan menjadi 4 komponen) dengan model RVFL terpisah per IMF, hasilnya dijumlahkan sebagai prediksi akhir (skema ensemble decomposition).
Perbedaan skema ini membuat kompleksitas model dan cara memanfaatkan informasi dekomposisi tidak identik.

Keempat, cutoff tanggal test set kami sengaja dipertahankan di 2020-02-10 (bukan menyamakan jumlah baris 8.596 pertama), karena jika dipaksakan menyamakan jumlah baris, test set akan mencakup 2020-04-20 dengan `actual_close` sebesar -36,98 (harga negatif kontrak berjangka WTI akibat krisis COVID-19), yang membuat MAPE tidak bermakna secara matematis untuk seluruh model.

## 5. Interpretasi

Dengan mempertimbangkan keempat faktor di atas, model terbaik kami pada run wu (TCN, MAPE 0,0224, Dstat 0,8918) masih jauh di bawah performa ICEEMDAN SCA RVFL milik Wu et al. (MAPE 0,0035, Dstat 0,9273) secara nilai mutlak.

Namun perbandingan yang lebih adil adalah terhadap baris "model tunggal" Wu et al. tanpa skema decomposition ensemble penuh (SCA RVFL saja, MAPE 0,0157, Dstat 0,7522), di mana model terbaik kami (TCN) justru unggul pada Dstat (0,8918 vs 0,7522) meski masih kalah pada MAPE.
Ini mengindikasikan sebagian besar keunggulan Wu et al. berasal dari skema decomposition ensemble per IMF plus look ahead leakage, bukan semata dari kapasitas model prediktif itu sendiri.

Setelah penalaan hyperparameter (Bagian 8 laporan utama), MLP Tuned (MAPE 0,0230, Dstat 0,8881) hampir menyamai TCN sebagai model terbaik kami, mengonfirmasi bahwa dengan konfigurasi tepat, arsitektur sederhana pun bisa kompetitif pada setup dataset ini, konsisten dengan temuan XAI bahwa sinyal utama (komponen Trend pada lag_1) sudah sangat informatif tanpa memerlukan arsitektur sequential yang kompleks.

Kesimpulannya, gap performa terhadap Wu et al. sebagian besar bersifat struktural (leakage dekomposisi dan skema ensemble per IMF), bukan murni indikasi model kami secara arsitektural lebih lemah.
Klaim "kalah" atau "menang" secara mentah terhadap Wu et al. tidak tepat tanpa konteks ini.
