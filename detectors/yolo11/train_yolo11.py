"""Обучение разных версий YOLO (v8, v11, v12) на вашем датасете."""

import sys
import tempfile
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.config import dataset

def train_yolo_version(model_name: str = "yolo11n.pt"):
    data_yaml_path = dataset.find_data_yaml()
    print(f"data.yaml: {data_yaml_path}")
    print(f"🚀 Обучение модели: {model_name}")

    with tempfile.TemporaryDirectory(prefix="yolo_data_") as tmpd:
        ultra_yaml = dataset.export_ultralytics_yaml_with_absolute_path(
            data_yaml_path, Path(tmpd) / "data_abs_path.yaml"
        )
        print(f"Абсолютный путь к data.yaml: {ultra_yaml}")

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
        
        try:
            model = YOLO(model_name)
        except Exception as e:
            print(f"Ошибка загрузки {model_name}")

        model.train(
            data=str(ultra_yaml),
            epochs=150,
            batch=8,
            imgsz=640,
            device=device,
            workers=4,
            amp=bool(torch.cuda.is_available()),
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            momentum=0.937,
            hsv_h=0.02,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            shear=5,
            perspective=0.0,
            flipud=0.3,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            single_cls=True,
            patience=15,
            save_period=10,
            plots=True,
            exist_ok=True, 
        )
        
        print(f"\n✅ Обучение {model_name} завершено.")
        print(f"Веса сохранены в runs/detect/ (см. папку с именем trainX или train)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение разных версий YOLO")
    parser.add_argument("--model", type=str, default="yolo11n.pt", 
                        help="Имя модели: yolov8n.pt, yolo11n.pt, yolo12n.pt, yolov12n.pt")
    args = parser.parse_args()
    
    train_yolo_version(args.model)