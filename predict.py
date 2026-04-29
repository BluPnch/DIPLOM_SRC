from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

input_dir = Path("test_img")
output_dir = Path("runs/predict_custom")
output_dir.mkdir(parents=True, exist_ok=True)

extensions = (".jpg")

for img_path in input_dir.iterdir():
    if img_path.suffix.lower() not in extensions:
        continue 

    print(f"\nОбработка: {img_path.name}")

    results = model.predict(source=str(img_path), conf=0.25, save=False)

    for result in results:
        num_objects = len(result.boxes) if result.boxes is not None else 0
        print(f"  -> {img_path.name}: {num_objects} объектов (слизней)")
        annotated_img = result.plot() 
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated_img)

print(f"\nВсе результаты сохранены в папку: {output_dir}")