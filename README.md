# Peramalan Harga Minyak Mentah WTI dengan Deep Learning dan XAI

Proyek tugas akhir mata kuliah Deep Learning, replikasi eksperimen `experiment-argument-ID.md` pada dataset hasil dekomposisi CEEMDAN, dilengkapi analisis XAI dan perbandingan terhadap penelitian acuan Wu et al. (ICEEMDAN SCA RVFL).

## Tujuan

Melatih delapan arsitektur deep learning (MLP, RNN, LSTM, BiLSTM, GRU, TCN, Transformer, Informer) untuk memprediksi harga penutupan WTI satu hari ke depan, memakai empat komponen hasil dekomposisi CEEMDAN sebagai input.
Menjelaskan perilaku model memakai SHAP dan Integrated Gradients.
Membandingkan performa terhadap Wu et al. secara jujur dan kontekstual.

Laporan lengkap ada di `final-task-report-ID.md`, perbandingan Wu et al. ada di `benchmark-wu-comparison.md`.

## Struktur Folder

Folder `dataset` berisi data sumber dan hasil split.
`dataset/WTI-CEEMDAN-FE-n10.csv` adalah dataset sumber.
`dataset/splits/{full,wu}/` berisi hasil windowing dan split (`splits.npz`, `split-report.md`).
`dataset/scalers/{full,wu}/` berisi scaler hasil fit pada train.

Folder `models/{full,wu}/` berisi checkpoint 8 model per run, plus checkpoint tambahan dari eksperimen lanjutan (`*_recency_fixed_model.pt`, `mlp_tuned_model.pt`).
Folder ini di-gitignore, checkpoint tidak ikut ter-commit ke repository.

Folder `evaluations/statistical/` dan `evaluations/graphical/` berisi hasil evaluasi dalam bentuk tabel markdown, CSV, dan plot, dipecah per tahap.

`model-train/{full,wu}/`, metrik training 8 model, uji Diebold Mariano, feature importance permutation.
`recency-fix/{full,wu}/`, hasil eksperimen mitigasi recency bias pada Transformer dan Informer.
`hyperparameter-tuning/{full,wu}/`, hasil penalaan hyperparameter MLP.

Folder `evaluations/xai/statistical/{full,wu}/` dan `evaluations/xai/graphical/{full,wu}/` berisi hasil analisis SHAP dan Integrated Gradients.

Folder `references/` berisi dokumen penelitian acuan (Wu et al.).
Folder `docs/superpowers/plans/` berisi rencana implementasi lengkap proyek.

## Cara Menjalankan Pipeline

Aktifkan virtual environment terlebih dahulu (Python 3.11 dengan torch CUDA, shap, dan captum terpasang), lalu jalankan script secara berurutan.

Tahap 1, persiapan fitur dan split data.

```bat
python 04-fe-and-split.py --run full
python 04-fe-and-split.py --run wu
```

Tahap 2, training 8 arsitektur.

```bat
python 05-dl-model-training.py --run full
python 05-dl-model-training.py --run wu
```

Tahap 3, analisis XAI (SHAP dan Integrated Gradients).

```bat
python 09-xai-explainability.py --run full
python 09-xai-explainability.py --run wu
```

Tahap 3b, eksperimen lanjutan mitigasi recency bias (Transformer dan Informer), dijalankan setelah Tahap 3 menunjukkan kedua model punya timestep concentration paling rendah.

```bat
python 10-recency-bias-fix.py --run full
python 10-recency-bias-fix.py --run wu
```

Tahap 3c, eksperimen lanjutan penalaan hyperparameter MLP, dijalankan setelah Tahap 3 menunjukkan MLP model terlemah tanpa kelemahan struktural.

```bat
python 11-hyperparameter-tuning.py --run full
python 11-hyperparameter-tuning.py --run wu
```

Setiap tahap wajib dijalankan untuk run `full` maupun `wu` sebelum melanjutkan ke tahap berikutnya, karena tahap XAI dan eksperimen lanjutan bergantung pada checkpoint model dari tahap training.

## Daftar Laporan

`final-task-report-ID.md`, laporan utama, mencakup dataset, delapan arsitektur, hasil training kedua run, hasil XAI, dan kedua eksperimen lanjutan.
`benchmark-wu-comparison.md`, perbandingan detail terhadap Wu et al., termasuk catatan kejujuran perbandingan.
`todo-list.md`, checklist progres implementasi proyek.
`docs/superpowers/plans/2026-07-13-final-task-dl.md`, rencana implementasi lengkap.
