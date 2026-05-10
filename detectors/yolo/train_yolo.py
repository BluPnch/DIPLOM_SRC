"""Обучение детектора YOLO (Ultralytics) на вашем датасете."""

import sys
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dataset_yaml  # noqa: E402

if __name__ == "__main__":
    data_yaml_path = dataset_yaml.find_data_yaml()
    print(f"data.yaml: {data_yaml_path}")

    with tempfile.TemporaryDirectory(prefix="yolo_data_") as tmpd:
        ultra_yaml = dataset_yaml.export_ultralytics_yaml_with_absolute_path(
            data_yaml_path, Path(tmpd) / "data_abs_path.yaml"
        )
        print(f"Для обучения Ultralytics: абсолютный path в {ultra_yaml}")

        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
        model = YOLO("yolo26n.pt")

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
            exist_ok=False,
        )

        print(
            "\n✅ Обучение завершено. Веса: runs/detect/.../weights/best.pt "
            "или при запуске из другой папки — см. каталог runs рядом с рабочей директорией."
        )
