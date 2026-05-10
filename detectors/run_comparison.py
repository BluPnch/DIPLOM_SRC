# detectors/run_comparison.py
import sys
from pathlib import Path

# Добавляем пути
sys.path.insert(0, str(Path(__file__).parent / "faster_rcnn"))

print("="*60)
print("Сравнение YOLO vs Faster R-CNN")
print("="*60)

# 1. Проверяем наличие данных
from faster_rcnn.prepare_data_for_faster_rcnn import convert_yolo_to_coco
print("\n[1/4] Конвертация данных в COCO формат...")
convert_yolo_to_coco()

# 2. Обучаем Faster R-CNN
print("\n[2/4] Обучение Faster R-CNN...")
from faster_rcnn.train_faster_rcnn_simple import train_faster_rcnn
faster_model = train_faster_rcnn()

# 3. Загружаем YOLO
print("\n[3/4] Загрузка YOLO модели...")
from ultralytics import YOLO

yolo_paths = [
    Path("runs/detect/train/weights/best.pt"),
    Path("../runs/detect/train/weights/best.pt"),
]

yolo_path = None
for path in yolo_paths:
    if path.exists():
        yolo_path = path
        break

if yolo_path:
    print(f"✅ Загружена обученная YOLO модель: {yolo_path}")
    yolo_model = YOLO(str(yolo_path))
else:
    print("⚠️ Обученная модель не найдена")

# 4. Сравнение
print("\n[4/4] Сравнение моделей...")
print("\n" + "="*60)

# Простое тестирование
import time
import numpy as np
from PIL import Image

test_image_path = Path("test_dataset/images/val")
if test_image_path.exists():
    test_images = list(test_image_path.glob("*.jpg"))[:5]
    
    print(f"\nТестирование на {len(test_images)} изображениях...")
    
    for model_name, model in [("YOLO", yolo_model), ("Faster R-CNN", faster_model)]:
        if model is None:
            continue
            
        times = []
        for img_path in test_images:
            start = time.time()
            
            if model_name == "YOLO":
                results = model(str(img_path), verbose=False)
                num_detections = len(results[0].boxes) if results[0].boxes else 0
            else:
                # Faster R-CNN
                from torchvision import transforms
                img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.ToTensor()(img).unsqueeze(0)
                with torch.no_grad():
                    predictions = model(img_tensor)
                num_detections = len(predictions[0]['boxes'])
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(f"{model_name:15} | Среднее время: {avg_time*1000:.1f}ms | FPS: {fps:.1f}")

print("\n✅ Сравнение завершено!")