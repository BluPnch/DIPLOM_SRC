"""Визуализация этапов предварительной обработки изображения для слайда"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend (no GUI)
import matplotlib.pyplot as plt
from pathlib import Path

# Параметры
TARGET_SIZE = 640
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "preprocessing_stages.png"

def create_test_image():
    """Создаёт синтетическое тестовое изображение"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), -1)
    cv2.rectangle(img, (250, 100), (400, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (450, 50), (600, 180), (255, 0, 0), -1)
    cv2.circle(img, (320, 300), 80, (255, 255, 0), -1)
    cv2.putText(img, "Slug", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (0, 0, 0), 3)
    return img

def preprocess_stages(image):
    """Выполняет три этапа предобработки"""
    stages = {}
    
    # 1. Исходное изображение
    stages['original'] = image.copy()
    
    # 2. Масштабирование до 640x640 (с сохранением пропорций + дополнение)
    h, w = image.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Добавляем рамку до 640x640
    padded = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 128
    y_offset = (TARGET_SIZE - new_h) // 2
    x_offset = (TARGET_SIZE - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    stages['resized'] = padded
    
    # 3. Нормализация (визуально не меняется)
    normalized = padded.astype(np.float32) / 255.0
    stages['normalized'] = (normalized * 255).astype(np.uint8)
    
    return stages

def create_comparison_grid(stages):
    """Создаёт сетку 1x3 для трёх этапов и сохраняет в файл"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = {
        'original': 'Исходное изображение',
        'resized': f'Масштабирование\nдо {TARGET_SIZE}×{TARGET_SIZE}',
        'normalized': 'Нормализация\n(÷255)'
    }
    
    for idx, (key, img) in enumerate(stages.items()):
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(titles[key], fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        h, w = img.shape[:2]
        axes[idx].text(0.5, -0.08, f'{w}×{h}', 
                      transform=axes[idx].transAxes,
                      ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Закрываем фигуру, не показывая
    print(f"Сохранено: {OUTPUT_FILE.absolute()}")

def main():
    image_path = Path(__file__).parent / "test_image.jpg"
    
    if image_path.exists():
        img = cv2.imread(str(image_path))
        print(f"Загружено изображение: {image_path.name}, размер: {img.shape[1]}×{img.shape[0]}")
    else:
        img = create_test_image()
        print("Создано синтетическое тестовое изображение")
    
    stages = preprocess_stages(img)
    create_comparison_grid(stages)
    
    # Проверяем, что файл создан
    if OUTPUT_FILE.exists():
        print(f"Файл успешно создан: {OUTPUT_FILE}")
        print(f"Размер файла: {OUTPUT_FILE.stat().st_size} байт")
    else:
        print("ОШИБКА: файл не был создан!")

if __name__ == "__main__":
    main()