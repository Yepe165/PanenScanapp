# Panen Scan — Panduan Training Model

Langkah dari nol sampai model siap dipakai di aplikasi.

## Langkah 1 — Kumpulkan foto
- Kelapa sawit: 800–1500 foto, variasi cahaya & jarak
- Buah/sayur lain: 500–1000 foto per jenis
- Sertakan variasi: latar polos & latar asli kebun, buah tunggal & bergerombol

## Langkah 2 — Anotasi (beri label)
Gunakan [Roboflow](https://roboflow.com) (gratis untuk skala kecil) atau [CVAT](https://cvat.ai):
1. Upload semua foto
2. Gambar kotak (bounding box) di tiap buah
3. Beri label kelas: `matang`, `mengkal`, atau `mentah`
4. Export dalam format **YOLOv8** — hasil export otomatis punya struktur folder
   `images/` dan `labels/` yang cocok dengan `dataset.yaml`

## Langkah 3 — Latih model deteksi
```bash
pip install ultralytics --break-system-packages
python train_detector.py
```
Model belajar mendeteksi lokasi tiap buah sekaligus kelas kematangannya.
Jumlah buah dalam satu foto = jumlah bounding box yang terdeteksi.

## Langkah 4 — Kumpulkan data berat (untuk model estimasi kg)
- Foto tiap buah dengan jarak kamera konsisten, lalu timbang beratnya
- Susun ke `data_berat.csv` (lihat format di dalam `train_weight_model.py`)

## Langkah 5 — Latih model estimasi berat
```bash
pip install scikit-learn pandas joblib --break-system-packages
python train_weight_model.py
```

## Langkah 6 — Deploy ke aplikasi mobile
- `train_detector.py` sudah otomatis export ke `.tflite` (ringan, jalan di Android/iOS)
- Untuk model berat (`model_berat.joblib`), convert ke ONNX atau tulis ulang
  rumusnya sebagai fungsi sederhana di app (karena modelnya kecil)
- Alur di aplikasi: foto → model deteksi (hitung jumlah + kematangan per buah)
  → tiap bounding box dikirim ke model berat → jumlahkan semua estimasi berat

## Target akurasi yang realistis
- Deteksi & klasifikasi kematangan: mAP50 > 0.85 sudah cukup baik untuk mulai uji lapangan
- Estimasi berat: e   rror < 10-15% dari berat asli biasanya cukup untuk kebutuhan
  perkiraan hasil panen (bukan untuk transaksi jual-beli presisi)

## Catatan penting
Estimasi berat dari foto 2D punya batas akurasi karena kamera tidak tahu jarak
sebenarnya kecuali dikalibrasi. Untuk hasil lebih akurat, pertimbangkan kamera
depth (LiDAR di HP tertentu) atau selalu sertakan objek referensi ukuran tetap
di setiap foto.
