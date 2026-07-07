"""
Training model deteksi & klasifikasi kematangan buah menggunakan YOLOv8.

Instalasi (sekali saja):
    pip install ultralytics --break-system-packages

Cara pakai:
    python train_detector.py

Setelah training selesai, model tersimpan di:
    runs/detect/panen-scan/weights/best.pt

Model ini sekaligus melakukan 2 hal dalam satu proses:
  1. Deteksi lokasi tiap buah (bounding box) -> dipakai untuk MENGHITUNG JUMLAH
  2. Klasifikasi kelas tiap box (matang/mengkal/mentah) -> KEMATANGAN
"""

from ultralytics import YOLO

def main():
    # Mulai dari model kecil yang sudah dilatih di dataset umum (transfer learning).
    # Pilihan model: yolov8n.pt (paling ringan, cocok untuk HP) s/m/l/x (makin besar makin akurat, makin berat)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,          # berhenti lebih awal kalau tidak ada peningkatan
        project="runs/detect",
        name="panen-scan",
        augment=True,         # augmentasi bawaan (rotasi, brightness, dll) — penting karena
                              # foto kebun punya variasi cahaya & sudut yang tinggi
    )

    # Evaluasi cepat di data validasi
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)

    # Export ke format ringan untuk deploy ke HP (opsional)
    model.export(format="tflite")   # hasil: best_saved_model/best_float16.tflite
    # Alternatif: format="onnx" jika target bukan Android/TFLite

if __name__ == "__main__":
    main()
