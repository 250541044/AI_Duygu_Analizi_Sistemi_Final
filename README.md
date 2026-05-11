# AI_Duygu_Analizi_Sistemi_Final
# 🤖 AI Face & Emotion Analytics System

Bu proje, **Fırat Üniversitesi Teknoloji Fakültesi Yazılım Mühendisliği** bünyesinde geliştirilen, derin öğrenme tabanlı bir gerçek zamanlı duygu analizi sistemidir.

## 🚀 Öne Çıkan Özellikler
- **Yüksek Performans:** ONNX Runtime optimizasyonu sayesinde **0.45 Saniye** (Inference Time) stabil gecikme süresi.
- **Dinamik Filtreleme:** Duygu değişimlerindeki titremeyi engelleyen **Sliding Window (Hafıza Filtresi)** algoritması.
- **Görüntü İşleme:** CLAHE algoritması ile değişken ışık koşullarında yüksek doğruluk.
- **Modern Dashboard:** FastAPI tabanlı, asenkron veri akışına sahip karanlık mod arayüz.

## 🛠️ Kullanılan Teknolojiler
- **Backend:** FastAPI (Python 3.11+)
- **Computer Vision:** OpenCV (Haar Cascade)
- **Deep Learning:** ONNX Runtime (FERPlus Model)
- **Frontend:** HTML5, CSS3, JavaScript

## 📁 Proje Yapısı
- `main.py`: Kamera akışını yöneten ve AI analizini yapan ana sunucu.
- `index.html`: Verileri anlık görselleştiren dashboard arayüzü.
- `Hafta 5-6 Raporları`: Projenin akademik gelişim süreçlerini içeren PDF dosyaları.
- `emotion-ferplus-8.onnx`: Duygu analizi için kullanılan eğitilmiş model.

## ⚙️ Kurulum
```bash
pip install fastapi uvicorn opencv-python onnxruntime numpy
python main.py```
