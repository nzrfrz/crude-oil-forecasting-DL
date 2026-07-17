# Design Spec, Task 5, Stacking Ensemble

Tanggal, 2026-07-17.
Status, disetujui menunggu review user sebelum masuk implementation plan.

## Latar Belakang

Task 0 sampai Task 4 (training 8 model dasar, XAI, mitigasi recency-bias untuk Transformer/Informer, hyperparameter tuning MLP, laporan akhir) sudah selesai dan ter-commit. Eksperimen lanjutan yang diminta, stacking ensemble, mengombinasikan prediksi beberapa model lewat meta-learner, mengikuti pola yang sudah pernah dicoba di repo referensi `D:\Coding\#bigdata\crude-oil-forecasting-DL\07-stacking-ensemble.py` (Stack-All5 dan Stack-Top3, meta-learner RidgeCV difit di validation set). Di repo referensi, stacking TIDAK jelas mengalahkan model individu terbaik, hasil itu jadi konteks ekspektasi realistis untuk eksperimen ini.

## Tujuan

Menjawab pertanyaan, apakah menggabungkan prediksi beberapa arsitektur lewat meta-learner linear menghasilkan performa lebih baik dari model individu terbaik, pada dataset CEEMDAN lokal proyek ini (bukan dataset referensi), di kedua run (`full` dan `wu`).

## Cakupan

Dua run, `full` dan `wu`, sama seperti semua task sebelumnya. Run `full` punya test set dan unseen set, run `wu` cuma test set (sesuai split yang sudah ada).

## Base Learner Pool

Delapan base learner, satu per arsitektur (MLP, RNN, LSTM, BiLSTM, GRU, TCN, Transformer, Informer). Untuk arsitektur yang punya lebih dari satu varian checkpoint (MLP baseline vs Tuned, Transformer baseline vs Fixed, Informer baseline vs Fixed), varian yang dipakai dipilih OTOMATIS berdasarkan MAE test terendah pada run yang sama. Pemilihan ini bisa berbeda antara run `full` dan run `wu`, dan WAJIB dicetak eksplisit di laporan (varian mana yang menang dan MAE pembandingnya) supaya keputusan pemilihan transparan, bukan diasumsikan "yang di-tuned pasti lebih baik".

Alasan pemilihan otomatis, bukan selalu pakai varian tuned/fixed, ditemukan saat eksplorasi bahwa Transformer-fix di run `full` justru sedikit lebih buruk di MAE test (3.5732 vs 3.5398 baseline) meski recency concentration membaik, sementara di run `wu` Transformer-fix lebih baik di semua metrik. Kalau dipaksa selalu pakai varian fixed, base learner pool run `full` berisiko memasukkan model yang secara metrik lebih lemah dari alternatifnya.

RNN, LSTM, BiLSTM, GRU, TCN tidak punya varian lain, selalu pakai checkpoint Stage 05 (`models/<run>/<nama>_model.pt`).

## Pencarian Kombinasi Terbaik

Alih-alih dua komposisi tetap, dilakukan pencarian menyeluruh (exhaustive search) atas semua kombinasi ukuran 3 sampai 5 dari 8 base learner, `C(8,3) + C(8,4) + C(8,5) = 56 + 70 + 56 = 182` kombinasi per run.

Untuk tiap kombinasi, dua meta-learner difit dan dievaluasi.

1. `RidgeCV`, alpha grid `[0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]`, leave-one-out CV, sama seperti repo referensi.
2. `SimpleAverage`, rata-rata tak berbobot prediksi anggota, baseline naif untuk menunjukkan apakah RidgeCV benar-benar belajar bobot yang bermakna dibanding rata-rata polos.

Total `182 x 2 = 364` fit dievaluasi per run.

Kriteria rangking, MAE pada VALIDATION SET (bukan test set), supaya proses pemilihan kombinasi tidak bocor ke test set. Test set tetap murni dipakai satu kali di akhir, cuma untuk melaporkan performa kombinasi pemenang, konsisten dengan prinsip test set sebagai evaluasi akhir yang sudah dipegang di seluruh task sebelumnya (04 sampai 11).

Kombinasi pemenang, satu entri (kombinasi anggota, jenis meta-learner) dengan MAE validation terendah di antara 364 entri. Kalau `SimpleAverage` menang atas `RidgeCV` untuk kombinasi yang sama, itu dilaporkan apa adanya (tidak dipaksa RidgeCV menang).

## Fitting Meta-Learner

Validation set direplikasi persis dari logika Stage 05 (`05-dl-model-training.py` baris 960-964), 10% ekor kronologis dari `X_train` penuh, `n_val = int(len(X_train) * 0.10)`, TIDAK overlap dengan test/unseen, TIDAK butuh file baru karena bisa dihitung ulang dari `dataset/splits/<run>/splits.npz` yang sudah ada (`X_train`, `y_train`).

Prediksi tiap base learner pada validation set dikumpulkan jadi matriks fitur, `RidgeCV` difit terhadap `y_val` (skala scaled, sebelum inverse transform, konsisten dengan cara base learner dilatih). Untuk tiap kombinasi (3 sampai 5 anggota), matriks fitur cuma memakai kolom prediksi anggota kombinasi itu. Meta-learner yang sudah difit lalu diterapkan ke prediksi base learner pada test set (dan unseen set untuk run `full`), tapi HANYA untuk kombinasi pemenang, bukan seluruh 364 entri (evaluasi test/unseen untuk semua kombinasi tidak perlu, satu-satunya tujuan test/unseen adalah melaporkan angka final pemenang).

## Loading Model

Base learner butuh definisi class arsitektur yang tepat untuk memuat `state_dict`, disalin verbatim dari script sumbernya, bukan didefinisikan ulang.

- MLP baseline, `MLPModelBaseline` dari `11-hyperparameter-tuning.py` (identik `MLPModel` di `05-dl-model-training.py`), input flat `X[:, -1, :]` (4 fitur, timestep terakhir saja, BUKAN seluruh window 10 hari).
- MLP Tuned, `MLPModelTunable` dari `11-hyperparameter-tuning.py`, konstruktor pakai hyperparameter terbaik per run (run `full`, `h1=256, h2=128, h3=32, dropout=0.007929413671136584`, run `wu`, `h1=256, h2=128, h3=32, dropout=0.03625603004646558`, diambil dari `evaluations/statistical/hyperparameter-tuning/<run>/01_mlp_tuning_summary.md`), input flat sama seperti baseline.
- RNN, LSTM, BiLSTM, GRU, TCN, class masing masing dari `05-dl-model-training.py`, input sequence penuh `(batch, 10, 4)`.
- Transformer/Informer baseline, class dari `05-dl-model-training.py`, sinusoidal PE plus mean pooling, input sequence penuh.
- Transformer/Informer Fixed, `TransformerModelFixed`/`InformerModelFixed` dari `10-recency-bias-fix.py`, learned PE plus last timestep pooling, input sequence penuh.

Checkpoint dimuat dengan `map_location` sesuai device aktif, `model.eval()`, tanpa retrain apa pun (semua checkpoint sudah ada di `models/{full,wu}/`).

## Evaluasi

Metrik skala USD asli (inverse `scaler_y`), MAE, MAPE%, SMAPE%, RMSE, DA%, MASE, R2, format tabel sama seperti `01_metrics_summary.md` Stage 05, dihitung untuk 8 model individu (varian terpilih) plus 1 baris kombinasi pemenang.

DM test (fungsi `diebold_mariano_test` disalin dari `05-dl-model-training.py` baris 589) antara kombinasi pemenang versus model individu terbaik (MAE test terendah di antara 8 base learner), untuk test set, dan untuk unseen set kalau run `full`.

Kalau meta-learner pemenang adalah `RidgeCV`, bobot Ridge per anggota kombinasi dilaporkan sebagai bentuk interpretasi kontribusi. Kalau pemenang `SimpleAverage`, dicatat bahwa bobot implisit sama rata. Tidak ada analisis SHAP/Integrated Gradients tambahan, scope XAI dianggap sudah cukup dari Task 3.

## Output

Script baru, `12-stacking-ensemble.py`, mengikuti struktur folder yang sama dengan task sebelumnya.

- `evaluations/statistical/stacking/{full,wu}/01_stacking_summary.md`, berisi, varian base learner terpilih beserta alasan (MAE pembanding), ringkasan pencarian (182 kombinasi x 2 meta-learner = 364 entri, kriteria rangking MAE validation), tabel Top 10 kombinasi (anggota, jenis meta-learner, MAE validation), detail kombinasi pemenang (anggota, jenis meta-learner, bobot Ridge kalau relevan), tabel metrik 8 model individu plus kombinasi pemenang pada test/unseen, hasil DM test.
- `evaluations/graphical/stacking/{full,wu}/`, bar chart perbandingan MAE/MAPE/DA semua entry (8 individu plus 1 pemenang), plot actual vs predicted untuk kombinasi pemenang.
- `models/{full,wu}/stacking_winner.pkl`, meta-learner terfit dari kombinasi pemenang, disimpan via `joblib` (kalau `RidgeCV`, simpan objek Ridge, kalau `SimpleAverage`, simpan daftar nama anggota saja karena tidak ada parameter terlatih), tidak menimpa checkpoint model manapun.

## Catatan Kejujuran

Hasil stacking dilaporkan apa adanya di `01_stacking_summary.md`, termasuk kalau ternyata TIDAK mengalahkan model individu terbaik (skenario yang cukup mungkin, berdasarkan hasil serupa di repo referensi, lihat Latar Belakang). Interpretasi jujur tentang kenapa stacking gagal atau berhasil (misalnya, base learner terlalu berkorelasi sehingga tidak banyak keragaman untuk digabung, atau kombinasi ukuran kecil justru lebih baik dari ukuran besar karena model lemah menambah noise) ditulis di bagian akhir summary, mengikuti pola kejujuran metodologis yang sudah dipakai di `final-report.md` Bagian 7.4-7.6.

Rangking berbasis MAE validation juga punya risiko metodologis sendiri, dengan 364 entri dites terhadap validation set yang relatif kecil (10% dari train), ada kemungkinan overfitting seleksi (kombinasi menang di validation belum tentu menang di test). Ini WAJIB disebutkan di summary sebagai batasan, bukan disembunyikan, dan jadi alasan kenapa Top 10 kombinasi ditampilkan (bukan cuma pemenang tunggal), supaya pembaca bisa menilai apakah pemenang menang tipis atau menang telak dibanding kombinasi lain.

## Update ke `final-report.md` dan `todo-list.md`

Setelah eksperimen selesai dan diverifikasi user, `todo-list.md` ditambah Task 5 dengan checklist yang sama gaya dengan Task 3b/3c, dan `final-report.md` ditambah bagian baru (bagian 8) yang merangkum hasil stacking, konsisten dengan format bagian bagian sebelumnya.

## Di Luar Scope

Tidak ada retrain base learner. Tidak ada kombinasi di luar rentang ukuran 3 sampai 5 (kombinasi ukuran 1-2 atau 6-8 tidak dicari). Tidak ada varian ensemble lain (stacking dua level, boosting, dsb). Tidak ada analisis XAI untuk model ensemble. Tidak ada perbandingan ulang terhadap Wu et al. untuk hasil stacking (Bagian 7 final-report.md sudah menjawab itu untuk model individu, menambah stacking ke perbandingan itu di luar scope task ini kecuali diminta terpisah).
