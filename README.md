# Situs Flask: Deteksi Status Diabetes dengan SVM (BRFSS 2015)

Antarmuka web sederhana untuk mencoba model SVM hasil notebook
`Deteksi_Diabetes_SVM_BRFSS2015_SMOTE.ipynb`.

> **Hanya untuk penelitian dan edukasi.** Hasilnya bukan diagnosis medis. Model dilatih pada data survei
> Amerika Serikat dan, pada eksperimen sebelumnya, sulit mendeteksi kelas prediabetes.

## 1. Siapkan model
1. Jalankan notebook sampai **Bagian 14A**. Sel itu membuat `svm_diabetes_app.joblib`.
2. Di Colab, hapus tanda komentar pada baris `files.download(...)` untuk mengunduhnya.
3. Letakkan berkas itu di folder `model/` (atau set variabel lingkungan `MODEL_PATH`).

## 2. Pasang dan jalankan
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # samakan versi scikit-learn dengan notebook
python app.py
```
Buka http://127.0.0.1:5000

**Versi scikit-learn harus sama** dengan yang dipakai saat melatih model (tercetak pada keluaran Bagian 14A).
Jika berbeda, model bisa gagal dimuat atau memberi peringatan.

## 3. Halaman dan penggunaan
- **Beranda** (`/`): pengenalan, cara kerja, indikator yang dipakai, dan hal yang perlu diketahui.
- **Periksa** (`/periksa`): isi 21 indikator lalu tekan *Prediksi*. Tombol "Isi cepat" menyediakan nilai awal dan satu contoh
  pengisian (bukan data pasien nyata).
- **Tentang penelitian** (`/tentang`): latar belakang, dataset, metode, model aktif, keterbatasan, dan rujukan.
- **API JSON:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
 "HighBP":1,"HighChol":1,"CholCheck":1,"BMI":32,"Smoker":0,"Stroke":0,"HeartDiseaseorAttack":0,
 "PhysActivity":1,"Fruits":1,"Veggies":1,"HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,
 "GenHlth":3,"MentHlth":0,"PhysHlth":0,"DiffWalk":0,"Sex":0,"Age":9,"Education":5,"Income":6}'
```
- `GET /health` memeriksa apakah model berhasil dimuat.

## 4. Mengubah isi
- Teks dan angka dataset: `templates/` dan konstanta `DATASET_STATS` di `app.py` (angka berasal dari eksplorasi data).
- Tampilan: `static/style.css`. Tidak ada pustaka atau font dari internet, sehingga berjalan offline.

## 5. Catatan penting
- Skor yang ditampilkan adalah **skor keputusan SVM**, bukan probabilitas.
- Label pilihan (usia, pendidikan, pendapatan) mengikuti kodebook BRFSS/dokumentasi dataset; periksa kembali
  terhadap dokumentasi dataset yang Anda pakai sebelum dipublikasikan.
- SMOTE hanya dipakai saat pelatihan; aplikasi memuat pipeline `StandardScaler` + SVM saja, sehingga tidak
  membutuhkan `imbalanced-learn`.
- Aplikasi ini memakai server pengembangan Flask (`debug=False`, hanya `127.0.0.1`). Untuk publikasi ke internet
  gunakan server produksi (misalnya gunicorn) dan tinjau aspek privasi data kesehatan.
