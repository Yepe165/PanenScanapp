# Panen Scan — PWA

## Isi folder ini
- `index.html` — aplikasi utama (sudah ditambahi dukungan PWA)
- `manifest.json` — identitas app (nama, ikon, warna) supaya bisa di-"Add to Home Screen"
- `service-worker.js` — bikin app tetap bisa dibuka walau sinyal jelek/offline
- `icon-192.png`, `icon-512.png`, `icon-512-maskable.png` — ikon aplikasi

## Cara pakai (paling gampang)
1. Upload SEMUA file di folder ini ke hosting statis apa saja, contoh gratis:
   - Netlify Drop (netlify.com/drop) — tinggal drag & drop folder ini
   - GitHub Pages
   - Vercel
   - Firebase Hosting
2. PWA WAJIB diakses lewat HTTPS (kecuali localhost) — semua opsi di atas otomatis HTTPS.
3. Buka link hosting itu di HP (Chrome/Safari) → akan muncul opsi
   "Tambahkan ke Layar Utama" / "Add to Home Screen" → aplikasi jadi seperti app native.

## Uji coba lokal di komputer dulu (opsional)
Karena service worker butuh server (tidak bisa dibuka langsung dari file://), jalankan:
```bash
python3 -m http.server 8000
```
lalu buka `http://localhost:8000` di browser.

## Update ikon
Ikon dibuat otomatis (bunga sawit sederhana warna hijau tema app). Kalau mau ganti
dengan logo sendiri, cukup timpa `icon-192.png`, `icon-512.png`, dan
`icon-512-maskable.png` dengan ukuran piksel yang sama (persegi).

## Menghubungkan ke model AI (dari README utama proyek)
File `index.html` ini masih prototipe tampilan. Untuk deteksi kematangan &
estimasi berat sungguhan, sambungkan ke model hasil training (`train_detector.py`,
`train_weight_model.py`) — bisa lewat model TFLite yang dijalankan di browser
(TensorFlow.js) atau lewat API backend yang memanggil model tersebut.
