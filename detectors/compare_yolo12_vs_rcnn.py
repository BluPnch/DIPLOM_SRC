# detectors/compare_yolo12_vs_rcnn.py
"""Сравнение YOLOv12 и Faster R-CNN на тестовом датасете"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataset_yaml


class YOLOv12_vs_FasterRCNN:
    """Сравнение YOLOv12 и Faster R-CNN"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.detectors_dir = self.project_root / "detectors"
        
        # ПУТИ ДЛЯ YOLOv12 (исправлено)
        self.yolo12_weights = self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
        
        # Альтернативные пути для YOLOv12
        self.yolo12_alternatives = [
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train2" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train3" / "weights" / "best.pt",
            self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
            self.detectors_dir / "yolo" / "runs" / "train" / "weights" / "best.pt",
        ]
        
        # Путь для Faster R-CNN
        self.frcnn_weights = self.detectors_dir / "faster_rcnn" / "faster_rcnn_best.pth"
        
        self.results = {}
    
    def find_yolo12_weights(self):
        """Поиск весов YOLOv12"""
        print("   Поиск весов YOLOv12...")
        for path in self.yolo12_alternatives:
            if path.exists():
                print(f"   ✅ Найдены веса YOLOv12: {path}")
                return path
        return None
    
    def get_test_data(self):
        """Получение тестовых изображений и меток"""
        data_cfg, yaml_path = dataset_yaml.load_data_cfg()
        
        if 'test' in data_cfg:
            test_dir = Path(data_cfg['test'])
        else:
            test_dir = dataset_yaml.resolve_split_images_dir(data_cfg, "val", yaml_path)
        
        test_images = list(dataset_yaml.iter_image_paths(test_dir))
        labels_dir = dataset_yaml.yolo_labels_dir(test_dir, "val" if 'test' not in data_cfg else "test")
        
        print(f"📁 Тестовых изображений: {len(test_images)}")
        return test_images, labels_dir
    
    def load_models(self):
        """Загрузка моделей YOLOv12 и Faster R-CNN"""
        print("\n[1/4] Загрузка моделей...")
        
        # YOLOv12
        yolo12_path = self.find_yolo12_weights()
        if yolo12_path is None:
            raise FileNotFoundError(
                f"❌ YOLOv12 веса не найдены.\n"
                f"Искали в:\n" + "\n".join(f"  - {p}" for p in self.yolo12_alternatives)
            )
        yolo12 = YOLO(str(yolo12_path))
        print(f"   ✅ YOLOv12 загружена")
        
        # Faster R-CNN
        if not self.frcnn_weights.exists():
            raise FileNotFoundError(
                f"❌ Faster R-CNN веса не найдены: {self.frcnn_weights}\n"
                f"Убедитесь, что модель обучена: python detectors/faster_rcnn/train_faster_rcnn_simple.py"
            )
        
        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2,
        )
        
        faster_rcnn = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=640,
            max_size=640
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        faster_rcnn.load_state_dict(torch.load(str(self.frcnn_weights), map_location=device))
        faster_rcnn.eval()
        faster_rcnn.to(device)
        print(f"   ✅ Faster R-CNN загружена")
        
        return yolo12, faster_rcnn
    
    def predict_yolo12(self, model, image_path, conf_threshold=0.25):
        """Предсказание YOLOv12"""
        results = model(image_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int) + 1
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        return {
            'boxes': torch.tensor(boxes) if len(boxes) > 0 else torch.zeros((0, 4)),
            'scores': torch.tensor(scores) if len(scores) > 0 else torch.zeros(0),
            'labels': torch.tensor(labels) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)
        }
    
    def predict_faster_rcnn(self, model, image_path, conf_threshold=0.5):
        """Предсказание Faster R-CNN"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        image = image.resize((640, 640), Image.Resampling.BILINEAR)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred = predictions[0]
        
        keep = pred['scores'] > conf_threshold
        boxes = pred['boxes'][keep].cpu()
        scores = pred['scores'][keep].cpu()
        labels = pred['labels'][keep].cpu()
        
        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    def load_ground_truth(self, label_path, image_size):
        """Загрузка ground truth"""
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0]) + 1
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        x1 = (xc - w/2) * image_size[0]
                        y1 = (yc - h/2) * image_size[1]
                        x2 = (xc + w/2) * image_size[0]
                        y2 = (yc + h/2) * image_size[1]
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        return {
            'boxes': torch.tensor(boxes) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }
    
    def evaluate_accuracy(self, model, model_type, test_images, labels_dir):
        """Оценка точности модели"""
        print(f"\n[2/4] Оценка точности {model_type}...")
        
        metric = MeanAveragePrecision()
        predictions = []
        targets = []
        
        total = len(test_images)
        for i, img_path in enumerate(test_images):
            if (i + 1) % 50 == 0:
                print(f"      Обработано {i+1}/{total} изображений...")
            
            with Image.open(img_path) as img:
                img_size = img.size
            
            if model_type == 'YOLOv12':
                pred = self.predict_yolo12(model, img_path)
            else:
                pred = self.predict_faster_rcnn(model, img_path)
            
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt = self.load_ground_truth(label_path, img_size)
            
            if len(pred['boxes']) == 0:
                pred['boxes'] = torch.zeros((0, 4))
                pred['scores'] = torch.zeros(0)
                pred['labels'] = torch.zeros(0, dtype=torch.int64)
            
            predictions.append(pred)
            targets.append(gt)
        
        metric.update(predictions, targets)
        results = metric.compute()
        
        return {
            'mAP@0.5': results['map_50'].item() * 100,
            'mAP@0.5:0.95': results['map'].item() * 100,
            'mAP@small': results.get('map_small', torch.tensor(0)).item() * 100,
            'mAP@medium': results.get('map_medium', torch.tensor(0)).item() * 100,
            'mAP@large': results.get('map_large', torch.tensor(0)).item() * 100,
            'Recall@100': results.get('mar_100', torch.tensor(0)).item() * 100,
        }
    
    def measure_speed(self, model, model_type, test_images, num_runs=50):
        """Измерение скорости инференса"""
        print(f"\n[3/4] Измерение скорости {model_type}...")
        
        if not test_images:
            return 0, 0
        
        test_img_path = test_images[0]
        
        # Warmup
        for _ in range(5):
            if model_type == 'YOLOv12':
                model(test_img_path, verbose=False)
            else:
                image = Image.open(test_img_path).convert('RGB')
                image = image.resize((640, 640))
                img_tensor = transforms.ToTensor()(image).unsqueeze(0)
                device = next(model.parameters()).device
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    model(img_tensor)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            
            if model_type == 'YOLOv12':
                model(test_img_path, verbose=False)
            else:
                image = Image.open(test_img_path).convert('RGB')
                image = image.resize((640, 640))
                img_tensor = transforms.ToTensor()(image).unsqueeze(0)
                device = next(model.parameters()).device
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    model(img_tensor)
            
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        return fps, avg_time
    
    def get_model_size(self, model_path):
        """Размер модели в MB"""
        if model_path and Path(model_path).exists():
            return Path(model_path).stat().st_size / (1024 * 1024)
        return 0
    
    def run_comparison(self):
        """Запуск полного сравнения"""
        print("\n" + "="*70)
        print("СРАВНЕНИЕ: YOLOv12 vs FASTER R-CNN".center(70))
        print("="*70)
        
        # Получение тестовых данных
        test_images, labels_dir = self.get_test_data()
        
        if len(test_images) == 0:
            print("❌ Нет тестовых изображений!")
            return None
        
        # Загрузка моделей
        yolo12, faster_rcnn = self.load_models()
        
        # Оценка точности
        yolo12_metrics = self.evaluate_accuracy(yolo12, 'YOLOv12', test_images, labels_dir)
        faster_metrics = self.evaluate_accuracy(faster_rcnn, 'Faster R-CNN', test_images, labels_dir)
        
        # Измерение скорости
        yolo12_fps, yolo12_time = self.measure_speed(yolo12, 'YOLOv12', test_images)
        faster_fps, faster_time = self.measure_speed(faster_rcnn, 'Faster R-CNN', test_images)
        
        # Размер моделей
        yolo12_path = self.find_yolo12_weights()
        yolo12_size = self.get_model_size(yolo12_path)
        faster_size = self.get_model_size(self.frcnn_weights)
        
        # Формирование результатов
        self.results = {
            'YOLOv12': {
                **yolo12_metrics,
                'FPS': yolo12_fps,
                'Speed (ms)': yolo12_time,
                'Size (MB)': yolo12_size
            },
            'Faster R-CNN': {
                **faster_metrics,
                'FPS': faster_fps,
                'Speed (ms)': faster_time,
                'Size (MB)': faster_size
            }
        }
        
        return self.results
    
    def print_results(self):
        """Вывод результатов в виде таблицы"""
        if not self.results:
            print("❌ Нет результатов для отображения")
            return
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ".center(80))
        print("="*80)
        
        # Таблица точности
        print("\n📊 ТОЧНОСТЬ:")
        print("-"*80)
        print(f"{'Метрика':<25} {'YOLOv12':>20} {'Faster R-CNN':>20} {'Разница':>15}")
        print("-"*80)
        
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Recall@100']
        for metric in metrics:
            yolo12_val = self.results['YOLOv12'][metric]
            faster_val = self.results['Faster R-CNN'][metric]
            diff = faster_val - yolo12_val
            diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            print(f"{metric:<25} {yolo12_val:>19.2f}% {faster_val:>19.2f}% {diff_str:>15}")
        
        # Таблица производительности
        print("\n⚡ СКОРОСТЬ И РАЗМЕР:")
        print("-"*80)
        print(f"{'Метрика':<25} {'YOLOv12':>20} {'Faster R-CNN':>20} {'Сравнение':>15}")
        print("-"*80)
        
        speedup = self.results['YOLOv12']['FPS'] / self.results['Faster R-CNN']['FPS']
        size_ratio = self.results['Faster R-CNN']['Size (MB)'] / self.results['YOLOv12']['Size (MB)']
        
        print(f"{'FPS':<25} {self.results['YOLOv12']['FPS']:>19.1f} {self.results['Faster R-CNN']['FPS']:>19.1f} {'YOLO быстрее в ' + str(round(speedup, 1)) + 'x':>15}")
        print(f"{'Время (ms)':<25} {self.results['YOLOv12']['Speed (ms)']:>19.1f} {self.results['Faster R-CNN']['Speed (ms)']:>19.1f} {'YOLO быстрее':>15}")
        print(f"{'Размер (MB)':<25} {self.results['YOLOv12']['Size (MB)']:>19.1f} {self.results['Faster R-CNN']['Size (MB)']:>19.1f} {'YOLO меньше в ' + str(round(size_ratio, 1)) + 'x':>15}")
        
        print("\n" + "="*80)
    
    def plot_results(self):
        """Визуализация результатов"""
        if not self.results:
            print("❌ Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = ['YOLOv12', 'Faster R-CNN']
        colors = ['#2E86AB', '#A23B72']
        
        # График 1: mAP@0.5
        map_vals = [self.results[m]['mAP@0.5'] for m in models]
        bars = axes[0].bar(models, map_vals, color=colors, alpha=0.8)
        axes[0].set_ylabel('mAP@0.5 (%)')
        axes[0].set_title('Точность детекции', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, map_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
        
        # График 2: FPS
        fps_vals = [self.results[m]['FPS'] for m in models]
        bars = axes[1].bar(models, fps_vals, color=colors, alpha=0.8)
        axes[1].set_ylabel('FPS (кадров/сек)')
        axes[1].set_title('Скорость инференса', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, fps_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{val:.1f}', ha='center', fontweight='bold', fontsize=11)
        
        # График 3: Размер модели
        size_vals = [self.results[m]['Size (MB)'] for m in models]
        bars = axes[2].bar(models, size_vals, color=colors, alpha=0.8)
        axes[2].set_ylabel('Размер (MB)')
        axes[2].set_title('Компактность модели', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, size_vals):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{val:.1f} MB', ha='center', fontweight='bold', fontsize=11)
        
        plt.suptitle('Сравнение детекторов: YOLOv12 vs Faster R-CNN', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.detectors_dir / "yolo12_vs_frcnn.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\n📊 График сохранен: {output_path}")
        plt.show()
    
    def save_csv(self):
        """Сохранение результатов в CSV"""
        output_path = self.detectors_dir / "yolo12_vs_frcnn.csv"
        
        data = []
        for model_name, metrics in self.results.items():
            for metric_name, value in metrics.items():
                data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"📁 Результаты сохранены: {output_path}")
    
    def print_conclusion(self):
        """Вывод заключения"""
        print("\n" + "="*70)
        print("ВЫВОДЫ".center(70))
        print("="*70)
        
        yolo12_map = self.results['YOLOv12']['mAP@0.5']
        faster_map = self.results['Faster R-CNN']['mAP@0.5']
        yolo12_fps = self.results['YOLOv12']['FPS']
        faster_fps = self.results['Faster R-CNN']['FPS']
        
        print(f"\n1. По точности (mAP@0.5):")
        if yolo12_map > faster_map:
            print(f"   ✅ YOLOv12 точнее на {yolo12_map - faster_map:.2f}%")
        else:
            print(f"   📌 Faster R-CNN точнее на {faster_map - yolo12_map:.2f}%")
        
        print(f"\n2. По локализации (mAP@0.5:0.95):")
        yolo12_map095 = self.results['YOLOv12']['mAP@0.5:0.95']
        faster_map095 = self.results['Faster R-CNN']['mAP@0.5:0.95']
        if yolo12_map095 > faster_map095:
            print(f"   ✅ YOLOv12 лучше на {yolo12_map095 - faster_map095:.2f}%")
        else:
            print(f"   📌 Faster R-CNN лучше на {faster_map095 - yolo12_map095:.2f}%")
        
        print(f"\n3. По скорости (FPS):")
        print(f"   ✅ YOLOv12 быстрее в {yolo12_fps / faster_fps:.1f} раз")
        
        print(f"\n4. По компактности:")
        yolo12_size = self.results['YOLOv12']['Size (MB)']
        faster_size = self.results['Faster R-CNN']['Size (MB)']
        print(f"   ✅ YOLOv12 меньше в {faster_size / yolo12_size:.1f} раз")
        
        print("\n" + "="*70)
        print("💡 РЕКОМЕНДАЦИЯ:")
        if yolo12_map > 90:
            print("   ✅ YOLOv12 показывает отличную точность для задачи детекции слизней")
        if yolo12_fps > faster_fps * 10:
            print("   ✅ Для систем реального времени → используйте YOLOv12")
        print("="*70)


if __name__ == "__main__":
    comparator = YOLOv12_vs_FasterRCNN()
    results = comparator.run_comparison()
    
    if results:
        comparator.print_results()
        comparator.plot_results()
        comparator.save_csv()
        comparator.print_conclusion()