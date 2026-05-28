import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.paths import DETECTORS_DIR, REPO_ROOT


class ConfidenceThresholdResearch:
    
    def __init__(self, model_path=None):

        self.project_root = REPO_ROOT
        self.research_dir = self.project_root / "investigation" / "yolo12" / "confidence_study"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        
        # Загрузка модели
        if model_path is None:
            # Пробуем найти обученную модель YOLOv12
            possible_paths = [
                DETECTORS_DIR / "yolo12" / "runs" / "train" / "weights" / "best.pt",
                REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
                DETECTORS_DIR / "yolo26" / "runs" / "train" / "weights" / "best.pt",
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
        
        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        print(f"Загрузка модели: {model_path}")
        self.model = YOLO(str(model_path))
        print("Модель загружена успешно")
        
        # Пороги для исследования
        self.thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
        self.results = {}
    
    def detect_image(self, image_path, conf_threshold):
        """
        Детекция на одном изображении с заданным порогом
        
        Args:
            image_path: путь к изображению
            conf_threshold: порог уверенности
            
        Returns:
            dict: результаты детекции
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            num_detections = len(boxes)
        else:
            boxes = np.array([])
            scores = np.array([])
            num_detections = 0
        
        return {
            'num_detections': num_detections,
            'boxes': boxes,
            'scores': scores,
            'avg_confidence': np.mean(scores) if len(scores) > 0 else 0,
            'max_confidence': np.max(scores) if len(scores) > 0 else 0,
            'min_confidence': np.min(scores) if len(scores) > 0 else 0
        }
    
    def draw_detections(self, image_path, boxes, scores, threshold, output_path):

        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Используем разные цвета в зависимости от уверенности
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Цвет в зависимости от уверенности
            if score >= 0.7:
                color = (0, 255, 0)      # Зелёный - высокая уверенность
            elif score >= 0.5:
                color = (255, 255, 0)    # Жёлтый - средняя уверенность
            else:
                color = (0, 165, 255)    # Оранжевый - низкая уверенность
            
            # Рисуем рамку
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Рисуем текст с уверенностью
            text = f"{score:.2f}"
            draw.rectangle([x1, y1-20, x1+40, y1], fill=color)
            draw.text((x1+2, y1-18), text, fill=(0, 0, 0))
        
        # Добавляем информацию о пороге
        draw.rectangle([10, 10, 300, 50], fill=(0, 0, 0, 180))
        draw.text((15, 15), f"Confidence threshold: {threshold}", fill=(255, 255, 255))
        draw.text((15, 32), f"Detected slugs: {len(boxes)}", fill=(255, 255, 255))
        
        img.save(output_path)
    
    def analyze_single_image(self, image_path):
        image_name = Path(image_path).stem
        print(f"\nАнализ изображения: {image_name}")
        print("="*60)
        
        # Создаём папку для результатов этого изображения
        img_result_dir = self.research_dir / image_name
        img_result_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем оригинальное изображение
        original_img = Image.open(image_path)
        original_img.save(img_result_dir / "0_original.jpg")
        
        # Анализ на разных порогах
        results = []
        for threshold in self.thresholds:
            print(f"   Обработка порога {threshold:.2f}...")
            
            # Детекция
            detection = self.detect_image(image_path, threshold)
            results.append({
                'threshold': threshold,
                'detections': detection['num_detections'],
                'avg_conf': detection['avg_confidence'],
                'max_conf': detection['max_confidence'],
                'min_conf': detection['min_confidence'],
                'boxes': detection['boxes'],
                'scores': detection['scores']
            })
            
            # Сохраняем изображение с рамками
            if detection['num_detections'] > 0:
                output_path = img_result_dir / f"{int(threshold*100):03d}_thr_{detection['num_detections']}slugs.jpg"
                self.draw_detections(
                    image_path, 
                    detection['boxes'], 
                    detection['scores'],
                    threshold,
                    output_path
                )
        
        self.results[image_name] = results
        return results
    
    def plot_results(self, image_name):
        if image_name not in self.results:
            print(f"Результаты для {image_name} не найдены")
            return
        
        results = self.results[image_name]
        thresholds = [r['threshold'] for r in results]
        detections = [r['detections'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График 1: Количество детекций vs порог
        axes[0, 0].plot(thresholds, detections, 'b-o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].set_title('Detection Count vs Confidence Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(thresholds, detections, alpha=0.3)
        
        # График 2: Средняя уверенность детекций vs порог
        avg_conf = [r['avg_conf'] for r in results]
        axes[0, 1].plot(thresholds, avg_conf, 'r-o', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Confidence Threshold')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_title('Average Detection Confidence vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # График 3: Потеря детекций при повышении порога
        loss_rate = []
        max_detections = detections[0] if detections else 1
        for d in detections:
            loss = (max_detections - d) / max_detections * 100 if max_detections > 0 else 0
            loss_rate.append(loss)
        
        axes[1, 0].plot(thresholds, loss_rate, 'g-o', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Confidence Threshold')
        axes[1, 0].set_ylabel('Detection Loss (%)')
        axes[1, 0].set_title('Detection Loss vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(thresholds, loss_rate, alpha=0.3, color='green')
        
        # График 4: Кумулятивное распределение уверенности
        all_scores = []
        for r in results:
            all_scores.extend(r['scores'])
        
        if all_scores:
            axes[1, 1].hist(all_scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Confidence Scores')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Confidence Threshold Analysis: {image_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.research_dir / f"{image_name}_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"График сохранён: {output_path}")
    
    def create_summary_table(self, image_name):
        if image_name not in self.results:
            print(f"Результаты для {image_name} не найдены")
            return
        
        results = self.results[image_name]
        
        print(f"\nСводная таблица для {image_name}")
        print("="*80)
        print(f"{'Порог':<10} {'Обнаружено':<15} {'Ср. уверенность':<20} {'Мин. уверенность':<20} {'Макс. уверенность':<20}")
        print("-"*80)
        
        for r in results:
            print(f"{r['threshold']:<10.2f} {r['detections']:<15} {r['avg_conf']:<20.4f} {r['min_conf']:<20.4f} {r['max_conf']:<20.4f}")
        
        print("="*80)
        
        # Сохраняем таблицу в CSV
        import csv
        csv_path = self.research_dir / f"{image_name}_summary.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Detections', 'Avg_Confidence', 'Min_Confidence', 'Max_Confidence'])
            for r in results:
                writer.writerow([r['threshold'], r['detections'], r['avg_conf'], r['min_conf'], r['max_conf']])
        
        print(f"Таблица сохранена: {csv_path}")
    
    def create_comparison_grid(self, image_path):
        image_name = Path(image_path).stem
        img_result_dir = self.research_dir / image_name
        
        # Собираем изображения для разных порогов
        thresholds_to_show = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, threshold in enumerate(thresholds_to_show):
            # Загружаем сохранённое изображение
            img_path = img_result_dir / f"{int(threshold*100):03d}_thr_*.jpg"
            img_files = list(img_result_dir.glob(f"{int(threshold*100):03d}_thr_*.jpg"))
            
            if img_files:
                img = Image.open(img_files[0])
                axes[idx].imshow(img)
                axes[idx].set_title(f'Threshold = {threshold}', fontsize=12)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f'No detections\nat {threshold}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'Threshold = {threshold}', fontsize=12)
                axes[idx].axis('off')
        

        for idx in range(len(thresholds_to_show), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Comparison of Detection Results at Different Thresholds\n{image_name}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.research_dir / f"{image_name}_comparison_grid.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сетка сравнения сохранена: {output_path}")
    
    def generate_report(self, image_name):
        if image_name not in self.results:
            print(f"Результаты для {image_name} не найдены")
            return
        
        results = self.results[image_name]
        report_path = self.research_dir / f"{image_name}_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПОРОГА УВЕРЕННОСТИ НА ДЕТЕКЦИЮ\n")
            f.write(f"Изображение: {image_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("РЕЗУЛЬТАТЫ ИЗМЕРЕНИЙ:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Порог':<10} {'Обнаружено':<15} {'Ср. уверенность':<20}\n")
            f.write("-"*60 + "\n")
            
            for r in results:
                f.write(f"{r['threshold']:<10.2f} {r['detections']:<15} {r['avg_conf']:<20.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("ВЫВОДЫ:\n")
            f.write("-"*60 + "\n")
            

            max_detections = max(r['detections'] for r in results)
            optimal_threshold = 0.3
            for r in results:
                if r['detections'] >= max_detections * 0.9:
                    optimal_threshold = r['threshold']
                    break
            
            f.write(f"1. Оптимальный порог для данного изображения: {optimal_threshold:.2f}\n")
            f.write(f"2. Максимальное количество обнаруженных объектов: {max_detections}\n")
            f.write(f"3. При пороге 0.5 количество детекций составляет: {results[thresholds.index(0.5)]['detections'] if 0.5 in thresholds else 'N/A'}\n")
            f.write(f"4. Рекомендуемый порог для практического использования: 0.25-0.35\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"Отчёт сохранён: {report_path}")
    
    def run_full_analysis(self, image_path):
        print("\n" + "="*60)
        print("НАЧАЛО ИССЛЕДОВАНИЯ ПОРОГА УВЕРЕННОСТИ")
        print("="*60)
        
        self.analyze_single_image(image_path)
        
        image_name = Path(image_path).stem
        
        self.create_summary_table(image_name)
        
        self.plot_results(image_name)
        
        self.create_comparison_grid(image_path)
        
        self.generate_report(image_name)
        
        print("\n" + "="*60)
        print(f"ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
        print(f"Результаты сохранены в: {self.research_dir}")
        print("="*60)


def main():
    # IMAGE_PATH = input("Введите путь к изображению для анализа: ").strip()
    IMAGE_PATH = "C:\\sem8\\VKR\\DIPLOM_SRC\\test-conf.jpg"

    if not Path(IMAGE_PATH).exists():
        print(f"Ошибка: файл {IMAGE_PATH} не найден")
        return
    
    # Запуск исследования
    researcher = ConfidenceThresholdResearch()
    researcher.run_full_analysis(IMAGE_PATH)


if __name__ == "__main__":
    main()