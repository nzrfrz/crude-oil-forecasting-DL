# Todo List, Final Task DL

Plan lengkap ada di `docs/superpowers/plans/2026-07-13-final-task-dl.md`. Semua perintah `python` dijalankan sendiri oleh user di cmd dengan venv aktif.

Aktivasi venv:

```bat
D:\Coding\#bigdata\venv\Scripts\activate.bat
```

## Task 0. Scaffold

- [ ] Verifikasi venv, `python -c "import torch, shap, captum; print(torch.cuda.is_available())"` harus `True`
- [ ] Buat `.gitignore`
- [ ] Buat folder skeleton `dataset/splits/{full,wu}`, `dataset/scalers/{full,wu}`, `models/{full,wu}`, `evaluations/statistical/model-train/{full,wu}`, `evaluations/graphical/model-train/{full,wu}`, `evaluations/xai/statistical/{full,wu}`, `evaluations/xai/graphical/{full,wu}`
- [ ] `git init` dan commit awal, commit TANPA baris co-author

## Task 1. Prep dan split

- [ ] Tulis `04-fe-and-split.py`
- [ ] User run, `python 04-fe-and-split.py --run full`
- [ ] User run, `python 04-fe-and-split.py --run wu` (cutoff 2020-02-10, ekspektasi 8.086 baris)
- [ ] Verifikasi, assert aditif lolos, shape X `(N, 10, 4)`, nilai dalam [0, 1], split report tertulis
- [ ] Commit

## Task 2. Training 8 model

- [ ] Adaptasi `05-dl-model-training.py` dari repo referensi, kelas model disalin verbatim, `N_FEATURES = 4`, loading dari npz, output metrik `.md`
- [ ] User run smoke test, `python 05-dl-model-training.py --run full --max-epochs 2`
- [ ] Hapus artefak smoke test, user run penuh, `python 05-dl-model-training.py --run full`
- [ ] User run, `python 05-dl-model-training.py --run wu`
- [ ] Verifikasi, 8 checkpoint `.pt`, `01_metrics_summary.md`, `02_dm_test.md`, plot
- [ ] Commit

## Task 3. XAI

- [ ] Adaptasi `09-xai-explainability.py`, SHAP GradientExplainer + Integrated Gradients, concentration score, Spearman vs MAE
- [ ] User run, `python 09-xai-explainability.py --run full`
- [ ] User run, `python 09-xai-explainability.py --run wu`
- [ ] Verifikasi, CSV atribusi fitur dan timestep, heatmap, comparison plot
- [ ] Commit

## Task 4. Laporan

- [ ] Tulis `final-task-report-ID.md`, angka diambil dari file hasil
- [ ] Tulis `benchmark-wu-comparison.md`, tabel vs Wu horizon 1, plus catatan kejujuran perbandingan
- [ ] Tulis `README.md`
- [ ] Lint tanda baca, tanpa dash naratif, kalimat ber ":" jadi list
- [ ] Commit

## Keputusan yang sudah terkunci

- Run `wu` memakai cutoff `Date <= 2020-02-10` (8.086 baris), BUKAN 8.596 baris pertama, karena 8.596 baris berakhir 2022-02-18 dan test window memuat harga negatif COVID (2020-04-20, −36.98) yang membuat MAPE tak bermakna
- Satu model per arsitektur, input `(batch, 10, 4)`, target `actual_close` langsung
- Commit git tanpa baris co-author
