"""
Training model estimasi BERAT (kg) tiap buah dari ukurannya di foto.

Data yang dibutuhkan: data_berat.csv dengan kolom:
    image_path, class_name, bbox_width_px, bbox_height_px, bbox_area_px,
    jarak_kamera_cm, berat_gram

Cara mengumpulkan data ini di lapangan:
  1. Foto tiap buah satu per satu dengan jarak kamera KONSISTEN (mis. 30cm),
     atau sertakan objek referensi (koin/penggaris) di setiap foto.
  2. Timbang buah tersebut dengan timbangan digital, catat beratnya.
  3. Jalankan model deteksi (train_detector.py) untuk dapat bbox_width_px,
     bbox_height_px, bbox_area_px dari tiap foto, gabungkan dengan berat asli.

Instalasi:
    pip install scikit-learn pandas joblib --break-system-packages
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    df = pd.read_csv("data_berat.csv")

    # Fitur: ukuran objek di foto (piksel) + jenis buah (karena densitas beda-beda)
    # + jarak kamera (kalau tidak konsisten, ini penting untuk normalisasi skala)
    feature_cols = ["bbox_width_px", "bbox_height_px", "bbox_area_px", "jarak_kamera_cm", "class_name"]
    target_col = "berat_gram"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("kelas", OneHotEncoder(handle_unknown="ignore"), ["class_name"])],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05))
    ])

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"Rata-rata selisih prediksi berat: {mae:.1f} gram")

    joblib.dump(pipeline, "model_berat.joblib")
    print("Model tersimpan sebagai model_berat.joblib")

if __name__ == "__main__":
    main()
