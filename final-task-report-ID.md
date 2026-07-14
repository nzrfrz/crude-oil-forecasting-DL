# Laporan Akhir, Peramalan Harga Minyak Mentah WTI dengan Deep Learning dan XAI

## 1. Pendahuluan dan Tujuan

Laporan ini merupakan replikasi dari rancangan eksperimen `experiment-argument-ID.md` pada dataset hasil dekomposisi CEEMDAN penuh, mencakup periode 1988 hingga 2026.
Tujuan utamanya ada tiga.

Pertama, melatih delapan arsitektur deep learning untuk memprediksi harga penutupan minyak mentah WTI satu hari ke depan, memakai empat komponen hasil dekomposisi CEEMDAN sebagai input.
Kedua, menjelaskan perilaku kedelapan model memakai teknik XAI (SHAP dan Integrated Gradients) untuk memahami fitur dan langkah waktu mana yang paling berkontribusi terhadap prediksi.
Ketiga, membandingkan performa model secara jujur dengan hasil penelitian acuan Wu et al. (metode ICEEMDAN SCA RVFL) pada rentang data yang sama.

Selain tiga tujuan inti tersebut, laporan ini juga mendokumentasikan dua eksperimen lanjutan yang muncul dari temuan XAI.
Eksperimen pertama adalah mitigasi recency bias pada Transformer dan Informer.
Eksperimen kedua adalah penalaan hyperparameter khusus MLP.

## 2. Dataset dan Preprocessing

Dataset sumber adalah `dataset/WTI-CEEMDAN-FE-n10.csv`, berisi 9.681 baris data harian dari 1988-01-11 sampai 2026-06-29.
Setiap baris memuat empat komponen hasil dekomposisi CEEMDAN.

Komponen tersebut adalah sebagai berikut.
Trend, komponen tren jangka panjang hasil smoothing LOWESS.
IMF Group 1, kelompok Intrinsic Mode Function frekuensi tinggi.
IMF Group 2, kelompok Intrinsic Mode Function frekuensi menengah.
Residual, sisa dekomposisi.

Setiap komponen memiliki fitur lag 1 sampai lag 10 dan target per komponen, serta kolom `actual_close` sebagai harga penutupan aktual.
Properti aditif dataset sudah diverifikasi, jumlah keempat target komponen sama dengan `actual_close`.

Dekomposisi CEEMDAN dijalankan dengan skema expanding window untuk mencegah look ahead leakage, konsekuensinya 500 baris pertama dari data mentah dipakai sebagai warmup dan tidak menghasilkan dekomposisi valid sehingga di-drop.
Ini sebabnya dataset final dimulai dari 1988-01-11, bukan dari awal data harga minyak mentah.

Keputusan desain input mengikuti `experiment-argument-ID.md` secara persis, satu model per arsitektur dengan input berbentuk `(batch, 10, 4)` (10 langkah waktu lookback, 4 komponen), dan target `actual_close` langsung.
Ini bukan skema per komponen terpisah seperti pada repo thesis referensi.

Dua run eksperimen dijalankan.

Run `full` memakai seluruh 9.681 baris dengan split 80/10/10 kronologis.
Hasilnya train 7.744 baris (1988-01-11 sampai 2018-09-24), test 968 baris (2018-09-25 sampai 2022-08-08), dan unseen 969 baris (2022-08-09 sampai 2026-06-29).
Setelah windowing lookback 10 hari, jumlah window training adalah 6.970, validasi 774, test 968, dan unseen 969.

Run `wu` memakai subset dengan cutoff `Date <= 2020-02-10` untuk mereplikasi rentang data penelitian Wu et al., menghasilkan 8.086 baris dengan split 80/20.
Hasilnya train 6.468 baris (1988-01-11 sampai 2013-08-28) dan test 1.618 baris (2013-08-29 sampai 2020-02-10).
Setelah windowing, jumlah window training adalah 5.822, validasi 646, dan test 1.618.

Cutoff `2020-02-10` dipilih secara sengaja, bukan memotong 8.596 baris pertama seperti jumlah sampel asli Wu et al.
Alasannya, jika dipaksakan 8.596 baris pertama, data akan berakhir di 2022-02-18 dan jendela test 80/20 akan mencakup tanggal 2020-04-20 dengan `actual_close` sebesar -36,98 (harga negatif akibat krisis kontrak berjangka WTI saat COVID-19).
Harga negatif membuat metrik MAPE tidak bermakna secara matematis (pembagi mendekati nol atau negatif), sehingga cutoff `2020-02-10` dipertahankan meski jumlah baris menjadi lebih sedikit dari Wu et al.

Preprocessing tambahan mencakup winsorization pada kuantil train (clip nilai ekstrem 1 persen dan 99 persen per fitur) dan scaling MinMaxScaler yang di-fit hanya pada data train untuk mencegah kebocoran informasi dari test dan unseen ke proses scaling.
Nilai fitur input berada pada rentang 0 sampai 1 setelah scaling.

Rentang kuantil winsorize berbeda tipis antara run full dan wu karena keduanya di-fit pada train masing-masing.
Sebagai catatan risiko, komponen Residual mengalami clipping cukup berat pada beberapa baris test dan unseen (38 sampai 52 persen baris), sudah diantisipasi sejak tahap perencanaan dan tidak mengubah kesimpulan utama laporan.

## 3. Delapan Arsitektur

Delapan arsitektur dilatih identik pada kedua run, diadaptasi verbatim dari repo referensi `crude-oil-forecasting-DL`.
Alasan pemilihan tiap arsitektur mengikuti rancangan pada `experiment-argument-ID.md`.

MLP (Multi Layer Perceptron), baseline non sequential.
Model ini hanya memakai fitur pada langkah waktu terakhir (t=0, lag_1), tanpa memori terhadap urutan waktu, sebagai pembanding paling sederhana.

RNN (Recurrent Neural Network) sederhana, model sequential paling dasar dengan memori jangka pendek melalui hidden state.

LSTM (Long Short Term Memory), varian RNN dengan gerbang lupa dan gerbang input untuk mengatasi vanishing gradient pada dependensi jangka panjang.

BiLSTM (Bidirectional LSTM), memproses urutan waktu dari dua arah untuk menangkap konteks maju dan mundur dalam window lookback.

GRU (Gated Recurrent Unit), varian RNN dengan gerbang lebih sederhana dari LSTM, sering kompetitif dengan biaya komputasi lebih rendah.

TCN (Temporal Convolutional Network), memakai dilated causal convolution, secara arsitektural memberi bobot besar pada observasi mendekati waktu prediksi lewat lapisan konvolusi terakhir.

Transformer, memakai mekanisme self attention dengan positional encoding sinusoidal dan mean pooling pada baseline Stage 05, dirancang untuk menangkap dependensi jangka panjang tanpa batasan recurrent.

Informer, varian Transformer dengan ProbSparse attention yang dirancang lebih efisien untuk sequence panjang, juga memakai sinusoidal positional encoding dan mean pooling pada baseline.

Semua model dilatih dengan prosedur identik, EarlyStopping dengan patience 20 epoch dan ReduceLROnPlateau, maksimum 200 epoch.

## 4. Hasil Training, Run Full

### 4.1. Metrik Test Set (968 window, 2018-09-25 sampai 2022-08-08)

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 3.1983 | 6.9060 | 6.4232 | 5.1681 | 80.48 | 0.4814 | 0.9440 |
| RNN | 2.7962 | 5.5286 | 5.5493 | 4.5414 | 86.78 | 0.4209 | 0.9567 |
| LSTM | 3.4650 | 6.8126 | 6.3931 | 6.3389 | 84.30 | 0.5215 | 0.9157 |
| BiLSTM | 3.0547 | 6.0119 | 5.8313 | 5.3297 | 85.95 | 0.4598 | 0.9404 |
| GRU | 3.1805 | 6.3740 | 6.3115 | 5.4059 | 84.71 | 0.4787 | 0.9387 |
| TCN | 3.1283 | 6.5207 | 6.0377 | 5.4572 | 84.40 | 0.4709 | 0.9375 |
| Transformer | 3.5398 | 6.4038 | 6.1244 | 5.9522 | 82.33 | 0.5328 | 0.9257 |
| Informer | 3.4012 | 6.7074 | 6.5110 | 5.5975 | 81.51 | 0.5119 | 0.9343 |

RNN adalah model terbaik pada test set full, MAPE 5,53 persen dan DA (directional accuracy) 86,78 persen.

### 4.2. Metrik Unseen Set (969 window, 2022-08-09 sampai 2026-06-29)

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 3.1099 | 3.7210 | 3.8767 | 6.2964 | 85.04 | 0.4978 | 0.6527 |
| RNN | 2.9908 | 3.6025 | 3.6411 | 5.9626 | 87.00 | 0.4787 | 0.6886 |
| LSTM | 3.2860 | 3.9014 | 3.8877 | 6.9208 | 86.38 | 0.5260 | 0.5805 |
| BiLSTM | 3.0320 | 3.6093 | 3.6477 | 6.2894 | 88.13 | 0.4853 | 0.6535 |
| GRU | 3.0896 | 3.6906 | 3.7163 | 6.3614 | 86.38 | 0.4945 | 0.6455 |
| TCN | 3.0933 | 3.7214 | 3.7641 | 6.2872 | 88.44 | 0.4951 | 0.6538 |
| Transformer | 3.3258 | 4.0147 | 4.0197 | 6.3566 | 83.08 | 0.5323 | 0.6461 |
| Informer | 3.5364 | 4.2453 | 4.2573 | 6.9664 | 80.39 | 0.5661 | 0.5749 |

Pada unseen set, MAPE semua model justru turun ke rentang 3,6 sampai 4,2 persen dan DA naik ke 80 sampai 88 persen, sekilas tampak lebih baik dari test set.
Namun R2 anjlok ke rentang 0,55 sampai 0,69, jauh di bawah R2 test set yang mencapai 0,92 sampai 0,96.

Penyebabnya bukan model memburuk, melainkan variansi target pada periode unseen jauh lebih kecil dari periode test.
Periode test (2018 sampai 2022) memuat crash harga akibat COVID-19 termasuk harga negatif kontrak berjangka pada April 2020, menghasilkan variansi harga sangat besar sehingga R2 tinggi lebih mudah dicapai secara relatif.
Periode unseen (2022 sampai 2026) relatif lebih stabil, variansi kecil membuat penyebut R2 (variansi total) kecil, sehingga rasio error terhadap variansi tampak lebih buruk meski error absolut (MAE, RMSE) sebenarnya serupa atau lebih kecil dari test set.

### 4.3. Diebold Mariano Test

Uji Diebold Mariano dijalankan pairwise untuk seluruh 8 model (28 pasangan, h=1, squared error loss, koreksi Harvey et al. 1997).

Pada test set, RNN secara statistik signifikan lebih baik dari seluruh 7 model lain (semua p<0,05, mayoritas p<0,0001), mengonfirmasi RNN sebagai model terbaik bukan hanya secara nilai metrik tapi juga signifikan secara statistik.
LSTM konsisten kalah signifikan dari MLP, RNN, BiLSTM, GRU, TCN, Transformer, dan Informer.
Transformer kalah signifikan dari Informer (DM=2,9256, p=0,0035).

Pada unseen set, pola berubah cukup drastis.
RNN masih lebih baik dari MLP secara signifikan (DM=2,5397, p=0,0113), tapi keunggulannya terhadap model lain tidak lagi seuniversal test set.
Yang menarik, Transformer justru jauh lebih baik dari Informer pada unseen set (DM=-7,3484, p<0,0001), berkebalikan dari urutan pada test set.
Informer secara umum kalah dari hampir semua model lain pada unseen set.

Perubahan urutan ranking antara test set dan unseen set ini adalah temuan penting, mengindikasikan model yang unggul pada satu rezim pasar (test set dengan crash COVID) belum tentu unggul pada rezim pasar lain (unseen set yang lebih stabil).

## 5. Hasil Training, Run Wu (Benchmark Setup)

### 5.1. Metrik Test Set (1.618 window, 2013-08-29 sampai 2020-02-10)

| Model | MAE | MAPE% | SMAPE% | RMSE | DA% | MASE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 2.4679 | 4.3933 | 4.4937 | 3.2932 | 81.89 | 0.5472 | 0.9722 |
| RNN | 1.2945 | 2.3309 | 2.3301 | 1.8954 | 88.57 | 0.2870 | 0.9908 |
| LSTM | 1.5556 | 2.7669 | 2.7419 | 2.0853 | 86.40 | 0.3449 | 0.9889 |
| BiLSTM | 1.3788 | 2.4473 | 2.4337 | 1.9594 | 87.70 | 0.3057 | 0.9902 |
| GRU | 1.4934 | 2.6634 | 2.6451 | 2.0638 | 86.46 | 0.3311 | 0.9891 |
| TCN | 1.2462 | 2.2432 | 2.2397 | 1.8762 | 89.18 | 0.2763 | 0.9910 |
| Transformer | 1.7047 | 2.9898 | 2.9936 | 2.2688 | 85.78 | 0.3780 | 0.9868 |
| Informer | 1.6921 | 2.9908 | 2.9802 | 2.2532 | 84.67 | 0.3752 | 0.9870 |

TCN adalah model terbaik pada run wu, MAPE 2,24 persen dan DA 89,18 persen, disusul ketat oleh RNN.
Uji Diebold Mariano mengonfirmasi TCN dan RNN tidak berbeda signifikan (DM=-0,7808, p=0,4350), keduanya sama sama menjadi kandidat model terbaik, sementara MLP kalah signifikan dari seluruh 7 model lain (semua p<0,0001).

Perbandingan detail terhadap hasil Wu et al. dijabarkan terpisah di `benchmark-wu-comparison.md`.

## 6. Hasil XAI, Run Full dan Wu

Analisis XAI memakai dua metode saling melengkapi, SHAP GradientExplainer dan Integrated Gradients, pada 200 sampel window per split (capped untuk menjaga waktu komputasi wajar mengingat jumlah window test dan unseen mencapai 968 sampai 1.618).

### 6.1. Feature Attribution

Komponen Trend (hasil smoothing LOWESS pada CEEMDAN) mendominasi atribusi SHAP di seluruh 8 model dan seluruh split.
Rentang dominasinya adalah 61 sampai 70 persen pada full test, 74 sampai 79 persen pada full unseen, dan 75 sampai 82 persen pada wu test.
Hasil Integrated Gradients menunjukkan pola serupa atau bahkan lebih ekstrem.

Permutation feature importance (dRMSE) mengonfirmasi urutan yang sama secara konsisten di kedelapan model dan ketiga split.
Trend paling penting (peningkatan RMSE 400 sampai 1.477 persen saat fitur diacak), disusul IMF Group 1 (50 sampai 280 persen), IMF Group 2 (25 sampai 130 persen), dan Residual mendekati nol atau bahkan negatif (menandakan Residual mendekati murni noise bagi model).

Temuan ini adalah konfirmasi kuat terhadap hipotesis leakage yang sudah dicurigai sejak analisis awal proyek.
Komponen Trend pada lag_1 (observasi paling dekat dengan target) kemungkinan sudah sangat mendekati nilai `actual_close` esok hari secara struktural, mengingat sifat smoothing LOWESS yang membuat komponen Trend berubah perlahan antar hari berdekatan.
Ini juga menjelaskan mengapa DA (directional accuracy) pada seluruh model jauh lebih tinggi (80 sampai 89 persen) dibanding eksperimen referensi dengan 7 fitur non dekomposisi (46 sampai 49 persen di sana), sinyal arah kemungkinan "bocor" dari proses dekomposisi itu sendiri, bukan murni kapasitas model.

Uji korelasi Spearman antara feature concentration (seberapa terpusat atribusi pada satu fitur, skala 0 sampai 1) dengan MAE tidak menunjukkan hasil signifikan di ketiga split.
Full test rho=-0,1429, p=0,7358.
Full unseen rho=0,1190, p=0,7789.
Wu test rho=-0,0476, p=0,9108.

Artinya, hipotesis awal bahwa "model dengan atribusi antar fitur lebih menyebar akan berkinerja lebih buruk" (dipinjam dari studi acuan dengan 7 fitur non dekomposisi) tidak terbukti pada setup CEEMDAN 4 komponen ini.
Feature concentration relatif seragam di seluruh arsitektur karena Trend selalu mendominasi, sehingga tidak ada variasi cukup besar antar model untuk menjelaskan perbedaan MAE.

### 6.2. Timestep Attribution

Berbeda dari feature concentration, timestep concentration (seberapa terpusat atribusi pada langkah waktu t=0/lag_1 terbaru dibanding 9 langkah waktu sebelumnya) menunjukkan korelasi negatif signifikan dengan MAE di ketiga split.

Full test rho=-0,7857, p=0,0362.
Full unseen rho=-0,8214, p=0,0234.
Wu test rho=-0,8571, p=0,0137.

Semua signifikan pada alpha 0,05, konsisten arah dan besaran di ketiga split.

TCN memiliki timestep concentration tertinggi (0,815 sampai 0,901 di ketiga split) dan konsisten menjadi salah satu model dengan performa terbaik.
RNN memiliki concentration menengah (0,357 sampai 0,415) dan juga berkinerja baik.
Sebaliknya, Transformer dan Informer memiliki timestep concentration terendah di antara model sequential (0,016 sampai 0,083 di ketiga split), atribusi menyebar relatif rata sepanjang 10 hari lookback, dan keduanya konsisten berkinerja terburuk di antara model sequential pada ketiga split.

Temuan ini adalah temuan utama bab XAI laporan, menggantikan hipotesis awal tentang feature concentration dengan versi yang lebih didukung data.
Bukan konsentrasi antar fitur yang membedakan leaderboard model, melainkan konsentrasi antar langkah waktu (recency bias).

Penjelasan mekanistiknya sebagai berikut.
Sinyal Trend yang paling informatif justru terletak pada lag_1 (observasi paling baru).
Model yang secara arsitektural memaksa bobot besar pada observasi terbaru cenderung menang.
TCN melakukan ini lewat lapisan dilated convolution terakhir yang paling dekat dengan output.
RNN melakukan ini secara alami lewat sifat vanishing gradient yang membuat kontribusi observasi lama meluruh.
Sebaliknya, model attention (Transformer dan Informer) dirancang menyebar perhatian ke seluruh window secara adaptif tanpa prior eksplisit terhadap recency, sehingga tidak secara otomatis mengutamakan sinyal Trend paling informatif di lag_1, dan akibatnya kalah dari model yang lebih "memaksa" recency bias.

Detail eksperimen mitigasi terhadap temuan ini dijabarkan pada Bagian 7.

## 7. Eksperimen Lanjutan 1, Mitigasi Recency Bias (Transformer dan Informer)

Berdasarkan temuan Bagian 6.2, dilakukan eksperimen mitigasi pada Transformer dan Informer, mengganti sinusoidal positional encoding dengan learned positional encoding, dan mengganti mean pooling dengan last timestep pooling (memakai representasi hanya pada t=0/lag_1 untuk prediksi akhir, bukan rata rata seluruh window).

Model varian "Fixed" dilatih dari nol dengan prosedur identik Stage 05 (EarlyStopping patience 20, ReduceLROnPlateau, maksimum 200 epoch), dibandingkan terhadap checkpoint baseline Stage 05 asli (tanpa dilatih ulang).

### 7.1. Run Full

**Transformer**

| Varian | MAE | MAPE% | RMSE | DA% | R2 | Timestep Concentration (test) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (test) | 3.5398 | 6.4038 | 5.9522 | 82.33 | 0.9257 | 0.0765 |
| Fixed (test) | 3.5732 | 6.9463 | 6.1536 | 82.64 | 0.9205 | 0.1625 |
| Baseline (unseen) | 3.3258 | 4.0147 | 6.3566 | 83.08 | 0.6461 | |
| Fixed (unseen) | 3.3878 | 4.0824 | 6.7120 | 82.35 | 0.6054 | |

**Informer**

| Varian | MAE | MAPE% | RMSE | DA% | R2 | Timestep Concentration (test) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (test) | 3.4012 | 6.7074 | 5.5975 | 81.51 | 0.9343 | 0.0743 |
| Fixed (test) | 2.9913 | 5.9306 | 4.6434 | 85.95 | 0.9548 | 0.1590 |
| Baseline (unseen) | 3.5364 | 4.2453 | 6.9664 | 80.39 | 0.5749 | |
| Fixed (unseen) | 3.3433 | 4.0490 | 6.2997 | 83.49 | 0.6524 | |

### 7.2. Run Wu

**Transformer**

| Varian | MAE | MAPE% | RMSE | DA% | R2 | Timestep Concentration (test) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.7047 | 2.9898 | 2.2688 | 85.78 | 0.9868 | 0.0340 |
| Fixed | 1.5877 | 2.8310 | 2.1191 | 86.53 | 0.9885 | 0.0965 |

**Informer**

| Varian | MAE | MAPE% | RMSE | DA% | R2 | Timestep Concentration (test) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1.6921 | 2.9908 | 2.2532 | 84.67 | 0.9870 | 0.0144 |
| Fixed | 1.4374 | 2.5063 | 2.0324 | 88.81 | 0.9894 | 0.1563 |

### 7.3. Interpretasi

Timestep concentration naik konsisten pada seluruh 4 kombinasi model dan run (Transformer full, Informer full, Transformer wu, Informer wu), mengonfirmasi bahwa learned positional encoding plus last timestep pooling berhasil memaksa model lebih fokus pada observasi terbaru, sesuai hipotesis mitigasi.

Namun perbaikan metrik akurasi tidak seragam.
Informer membaik jelas dan konsisten di seluruh 4 skenario (full test, full unseen, wu test), MAPE turun dari 6,71 menjadi 5,93 persen (full test), dan dari 2,99 menjadi 2,51 persen (wu test), R2 naik di semua skenario.
Transformer membaik pada wu test (MAPE 2,99 menjadi 2,83 persen) tapi sedikit memburuk pada full test dan full unseen (MAPE naik tipis, R2 turun tipis).

Efek mitigasi pada proyek ini secara umum lebih moderat dari eksperimen acuan pada studi referensi 7 fitur (yang melaporkan penurunan MAE Transformer dari 8,21 menjadi 2,63, penurunan sekitar 68 persen).
Penjelasannya konsisten dengan temuan Bagian 6.1, komponen Trend sudah mendominasi 60 sampai 82 persen atribusi bahkan pada model yang belum diperbaiki, mengindikasikan sinyal utama kemungkinan sudah "bocor" lewat proyeksi linear per langkah waktu pada input, bukan semata lewat mekanisme pooling.
Perbaikan pooling saja tidak cukup mengubah sumber sinyal, sehingga efeknya terasa tapi tidak sebesar saat sumber sinyal lebih tersebar merata di semua fitur.

## 8. Eksperimen Lanjutan 2, Penalaan Hyperparameter (MLP)

Analisis XAI pada Bagian 6 tidak dapat mendiagnosis MLP karena MLP tidak memiliki lookback window (hanya memakai fitur pada t=0), sehingga tidak ada timestep attribution untuk dianalisis.
MLP juga konsisten menjadi salah satu model terlemah pada kedua run (full test MAPE 6,91 persen, wu test MAPE 4,39 persen, terburuk di kedua run pada Diebold Mariano test).

Karena kelemahan MLP tidak bersifat struktural seperti Transformer/Informer (bukan soal recency bias atau attention), melainkan murni soal kapasitas dan regularisasi jaringan feedforward, MLP menjadi kandidat tepat untuk penalaan hyperparameter.
Keputusan mempersempit scope tuning hanya ke MLP (bukan seluruh 8 model) didasarkan pada temuan repo referensi, tuning generik pada Transformer dan Informer justru memperburuk performa (dimensi per head attention menjadi terlalu kecil dan overfit pada validation set kecil), sementara tuning MLP berhasil signifikan.

Penalaan memakai Optuna dengan TPE sampler dan MedianPruner, 30 trial, mencari kombinasi h1, h2, h3 (ukuran hidden layer), dropout, learning rate, dan weight decay.
Checkpoint hasil tuning disimpan terpisah dari checkpoint Stage 05 baseline, sehingga baseline tetap dapat direproduksi.

### 8.1. Run Full

Hyperparameter terbaik, h1=256, h2=128, h3=32, dropout=0,00793, lr=0,00111, weight_decay=0,0001.

| Varian | MAE | MAPE% | RMSE | DA% | R2 |
|---|---:|---:|---:|---:|---:|
| Baseline (test) | 3.1983 | 6.9060 | 5.1681 | 80.48 | 0.9440 |
| Tuned (test) | 2.6613 | 6.0406 | 4.6811 | 87.71 | 0.9540 |
| Baseline (unseen) | 3.1099 | 3.7210 | 6.2964 | 85.04 | 0.6527 |
| Tuned (unseen) | 2.8475 | 3.4375 | 5.8641 | 88.34 | 0.6988 |

### 8.2. Run Wu

Hyperparameter terbaik, h1=256, h2=128, h3=32, dropout=0,03626, lr=0,00165, weight_decay=0,0001.

| Varian | MAE | MAPE% | RMSE | DA% | R2 |
|---|---:|---:|---:|---:|---:|
| Baseline (test) | 2.4679 | 4.3933 | 3.2932 | 81.89 | 0.9722 |
| Tuned (test) | 1.2938 | 2.2953 | 1.9330 | 88.81 | 0.9904 |

### 8.3. Interpretasi

Penalaan hyperparameter berhasil memperbaiki MLP secara signifikan pada kedua run dan seluruh split.
Pada run full, MAPE test turun dari 6,91 menjadi 6,04 persen dan DA naik dari 80,48 menjadi 87,71 persen.
Pada run wu, perbaikannya jauh lebih besar, MAPE test turun dari 4,39 menjadi 2,30 persen (turun hampir separuh), hampir menyamai model terbaik run wu (TCN, MAPE 2,24 persen).

Hasil ini mengonfirmasi diagnosis Bagian 6.1 dan Bagian 8, kelemahan MLP memang murni soal kapasitas dan regularisasi, bukan keterbatasan arsitektural yang tidak dapat diatasi lewat tuning (berbeda dengan Transformer/Informer yang justru memburuk saat ditala secara generik, sesuai temuan repo referensi).
Kombinasi hidden layer lebih besar (256, 128, 32) dibanding baseline Stage 05 dan learning rate lebih tinggi tampak menjadi faktor utama perbaikan.

## 9. Kesimpulan

Delapan arsitektur deep learning berhasil dilatih pada dataset CEEMDAN 4 komponen untuk memprediksi harga penutupan WTI satu hari ke depan, pada dua rentang data (full 1988 sampai 2026, dan wu 1988 sampai 2020 mereplikasi rentang penelitian acuan).

RNN adalah model terbaik pada run full test set (MAPE 5,53 persen), sementara TCN terbaik pada run wu test set (MAPE 2,24 persen), keduanya model sequential sederhana yang secara arsitektural memprioritaskan observasi terbaru.
Model attention (Transformer, Informer) konsisten berkinerja lebih lemah di antara model sequential pada kedua run, dan MLP (non sequential) menjadi model terlemah secara keseluruhan sebelum tuning.

Analisis XAI mengungkap dua temuan penting.
Pertama, komponen Trend hasil dekomposisi CEEMDAN mendominasi atribusi (61 sampai 82 persen) di seluruh model dan split, mengindikasikan sinyal arah kemungkinan bocor lewat proses dekomposisi, menjelaskan DA yang jauh lebih tinggi dari eksperimen referensi non dekomposisi.
Kedua, dan menjadi temuan utama laporan ini, timestep concentration (bukan feature concentration) berkorelasi negatif signifikan dengan MAE di ketiga split (rho -0,79 sampai -0,86), model yang memaksa bobot besar pada observasi terbaru secara konsisten menang.

Eksperimen mitigasi recency bias pada Transformer dan Informer berhasil menaikkan timestep concentration di seluruh skenario, dengan perbaikan akurasi konsisten pada Informer namun campuran pada Transformer, mengindikasikan sumber sinyal sudah bocor lewat proyeksi linear input, bukan semata lewat mekanisme pooling.

Eksperimen penalaan hyperparameter pada MLP berhasil signifikan pada kedua run, mengonfirmasi kelemahan MLP bersifat kapasitas/regularisasi, bukan struktural.

Perbandingan terhadap Wu et al. menunjukkan performa model kami masih jauh di bawah secara nilai mutlak, namun perbandingan ini tidak apple to apple karena beberapa faktor struktural yang dijabarkan lengkap di `benchmark-wu-comparison.md`, terutama look ahead leakage pada dekomposisi Wu et al. yang dilakukan sekali pada seluruh series termasuk data test.
