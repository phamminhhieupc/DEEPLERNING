# YOLOv8 ile Baret (Hard Hat) Tespiti

Bu projede, inşaat sahalarında çalışan işçilerin **baret takıp takmadığını tespit eden** bir nesne tespit modeli geliştirdim.  
Model, Roboflow üzerinden alınan **Hard Hat Workers** veri seti kullanılarak YOLOv8 ile eğitilmiştir.

---

## 📌 Proje Özeti

- Roboflow'dan baret / işçi görüntüleri içeren veri seti indirildi.
- Veri seti çok büyük olduğu için, eğitim süresini kısaltmak amacıyla **küçük bir alt veri seti (`dataset_small`)** oluşturuldu.
- YOLOv8 (Ultralytics) kullanılarak model eğitildi.
- Eğitim sonrası model, valid setten seçilen bir görüntü üzerinde test edildi.
- Tahmin sonuçları `runs/detect/` klasörü altında saklandı.

Bu proje, CV’de göstermek üzere **uçtan uca bir bilgisayarla görü projesi** (veri → eğitim → tahmin → görsel çıktı) sunmaktadır.

---

## 🧱 Kullanılan Teknolojiler

- Python 3.10
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Roboflow (veri seti yönetimi)
- PyTorch
- OpenCV (isteğe bağlı, gerçek zamanlı demo için kullanılabilir)
- VS Code

---

## 📂 Proje Yapısı

```text
proje24_yolo_hardhat/
├─ configs/
│  └─ data_small.yaml          # YOLO veri seti konfig dosyası
├─ dataset_small/              # Küçültülmüş eğitim veri seti
│  ├─ train/
│  ├─ valid/
│  └─ test/
├─ models/
│  └─ best.pt                  # Eğitilmiş YOLOv8 model ağırlıkları (lokal)
├─ runs/
│  └─ detect/                  # Eğitim / tahmin çıktıları (lokal, git'e dahil değil)
├─ src/
│  ├─ train.py                 # YOLOv8 eğitim scripti
│  └─ test_image.py            # Tek bir görüntüde tahmin yapan script
├─ utils/
│  └─ create_small_dataset.py  # Büyük veri setinden küçük alt küme oluşturan yardımcı script
├─ README.md

🔧 Kurulum
# Sanal ortam (opsiyonel ama tavsiye edilir)
python -m venv venv
venv\Scripts\activate  # Windows

# Gerekli paketler
pip install ultralytics
pip install opencv-python  # (sadece kamera demosu/ekstra işler için)

🧪 Veri Seti

Proje, Roboflow'da yayınlanan Hard Hat Workers veri setini kullanmaktadır.

Roboflow linki:
https://universe.roboflow.com/

Klasik YOLOv8 klasör yapısına uygun olacak şekilde:

dataset_small/
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
|__ .gitignore

Veri seti yolları configs/data_small.yaml dosyasında tanımlanmıştır:

train: dataset_small/train/images
val: dataset_small/valid/images
test: dataset_small/test/images
nc: 1
names: ['head']

🏋️‍♀️ Model Eğitimi

src/train.py dosyası, YOLOv8 modelini eğitim için kullanır.

Örnek train.py akışı:

from ultralytics import YOLO

MODEL_NAME = "yolov8s.pt"      # veya yolov8n.pt
DATA_CONFIG = "configs/data_small.yaml"

def main():
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_CONFIG,
        epochs=25,
        imgsz=640,
        batch=16
    )

if __name__ == "__main__":
    main()


Terminalden çalıştırmak için:

cd src
python train.py

🔍 Tek Görüntü Üzerinde Test

src/test_image.py, eğitilen modeli kullanarak tek bir görüntü üzerinde tahmin yapar.