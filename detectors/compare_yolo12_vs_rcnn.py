# detectors/compare_yolo12_vs_rcnn.py
"""Детальное сравнение YOLOv12 и Faster R-CNN на тестовой выборке с вариацией порога"""

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


class DetailedModelComparator:
    """Детальное сравнение YOLOv12 и Faster R-CNN с вариацией порога уверенности"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.detectors_dir = self.project_root / "detectors"
        
        # Пути к моделям
        self.yolo12_weights = self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt"
        
        self.yolo12_alternatives = [
            self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
        
        self.frcnn_weights = self.detectors_dir / "faster_rcnn" / "faster_rcnn_best.pth"
        
        # Равные промежутки для порога уверенности
        self.confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Для хранения информации о каждой детекции
        self.detection_details = {
            'YOLOv12': [],
            'Faster R-CNN': []
        }
        
        self.results = {}
        self.detailed_results = {}
    
    def find_yolo12_weights(self):
        for path in self.yolo12_alternatives:
            if path.exists():
                print(f"   Найдены веса YOLOv12: {path}")
                return path
        return None
    
    def get_test_data(self):
        """Получение тестовых изображений и меток"""
        test_images_dir = Path("C:/sem8/VKR/DIPLOM_SRC/dataset/images/test")
        test_labels_dir = Path("C:/sem8/VKR/DIPLOM_SRC/dataset/labels/test")
        
        if not test_images_dir.exists():
            print(f"Папка с тестовыми изображениями не найдена: {test_images_dir}")
            return [], None
        
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.png"))
        test_images = sorted(test_images)
        
        print(f"Тестовых изображений: {len(test_images)}")
        print(f"Путь к тестовым изображениям: {test_images_dir}")
        print(f"Путь к разметке: {test_labels_dir}")
        
        return test_images, test_labels_dir
    
    def load_models(self):
        print("\n[1/4] Загрузка моделей...")
        
        # YOLOv12
        yolo12_path = self.find_yolo12_weights()
        if yolo12_path is None:
            raise FileNotFoundError("YOLOv12 веса не найдены")
        yolo12 = YOLO(str(yolo12_path))
        print(f"   YOLOv12 загружена")
        
        # Faster R-CNN
        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        
        backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT').features
        backbone.out_channels = 576
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
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
        print(f"   Faster R-CNN загружена")
        
        return yolo12, faster_rcnn
    
    def predict_yolo12(self, model, image_path, conf_threshold):
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
            'labels': torch.tensor(labels) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64),
            'num_detections': len(boxes),
            'confidence_scores': scores
        }
    
    def predict_faster_rcnn(self, model, image_path, conf_threshold):
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
            'labels': labels,
            'num_detections': len(boxes),
            'confidence_scores': scores.numpy() if len(scores) > 0 else np.array([])
        }
    
    def load_ground_truth_yolo(self, label_path, image_size):
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
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'num_gt': len(boxes)
        }
    
    def evaluate_at_threshold(self, model, model_type, test_images, labels_dir, conf_threshold):
        """Оценка модели при заданном пороге уверенности (с усреднением по 3 прогонам)"""
        
        num_runs = 3
        
        all_results = []
        all_confidences = []
        
        for run in range(num_runs):
            print(f"      Прогон {run+1}/{num_runs}...")
            
            metric = MeanAveragePrecision()
            predictions = []
            targets = []
            total_detections = 0
            total_gt = 0
            run_confidences = []
            
            for img_path in test_images:
                with Image.open(img_path) as img:
                    img_size = img.size
                
                if model_type == 'yolo':
                    pred = self.predict_yolo12(model, img_path, conf_threshold)
                else:
                    pred = self.predict_faster_rcnn(model, img_path, conf_threshold)
                
                # Сохраняем уверенности для вычисления средней точности
                if len(pred['confidence_scores']) > 0:
                    run_confidences.extend(pred['confidence_scores'])
                
                label_path = labels_dir / f"{img_path.stem}.txt"
                gt = self.load_ground_truth_yolo(label_path, img_size)
                
                if len(pred['boxes']) == 0:
                    pred['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    pred['scores'] = torch.zeros(0, dtype=torch.float32)
                    pred['labels'] = torch.zeros(0, dtype=torch.int64)
                
                predictions.append(pred)
                targets.append(gt)
                total_detections += pred['num_detections']
                total_gt += gt['num_gt']
            
            metric.update(predictions, targets)
            results = metric.compute()
            
            all_results.append({
                'mAP@0.5': results['map_50'].item() * 100,
                'mAP@0.5:0.95': results['map'].item() * 100,
                'Recall@100': results.get('mar_100', torch.tensor(0)).item() * 100,
                'total_detections': total_detections,
                'total_gt': total_gt
            })
            all_confidences.extend(run_confidences)
        
        # Усреднение результатов по 3 прогонам
        avg_results = {
            'threshold': conf_threshold,
            'mAP@0.5': np.mean([r['mAP@0.5'] for r in all_results]),
            'mAP@0.5:0.95': np.mean([r['mAP@0.5:0.95'] for r in all_results]),
            'Recall@100': np.mean([r['Recall@100'] for r in all_results]),
            'total_detections': int(np.mean([r['total_detections'] for r in all_results])),
            'total_gt': int(np.mean([r['total_gt'] for r in all_results])) if all_results[0]['total_gt'] > 0 else 0,
            'avg_confidence': np.mean(all_confidences) if len(all_confidences) > 0 else 0
        }
        
        return avg_results
    
    def measure_speed_at_threshold(self, model, model_type, test_images, conf_threshold, num_runs=30):
        if not test_images:
            return 0, 0
        
        test_img_path = test_images[0]
        
        # Warmup
        for _ in range(3):
            if model_type == 'yolo':
                model(test_img_path, conf=conf_threshold, verbose=False)
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
            
            if model_type == 'yolo':
                model(test_img_path, conf=conf_threshold, verbose=False)
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
        if model_path and Path(model_path).exists():
            return Path(model_path).stat().st_size / (1024 * 1024)
        return 0
    
    def run_detailed_comparison(self):
        print("\n" + "="*80)
        print("ДЕТАЛЬНОЕ СРАВНЕНИЕ: YOLOv12 vs FASTER R-CNN".center(80))
        print("НА ТЕСТОВОЙ ВЫБОРКЕ".center(80))
        print("="*80)
        
        test_images, labels_dir = self.get_test_data()
        
        if len(test_images) == 0:
            print("Нет тестовых изображений!")
            return None
        
        yolo12, faster_rcnn = self.load_models()
        
        yolo12_path = self.find_yolo12_weights()
        yolo12_size = self.get_model_size(yolo12_path)
        faster_size = self.get_model_size(self.frcnn_weights)
        
        yolo_results = []
        faster_results = []
        
        print("\n[2/4] Оценка точности при разных порогах (усреднение по 3 прогонам)...")
        print("-" * 70)
        
        for threshold in self.confidence_thresholds:
            print(f"\n   Порог {threshold}:")
            
            print(f"      YOLOv12...")
            yolo_metrics = self.evaluate_at_threshold(
                yolo12, 'yolo', test_images, labels_dir, threshold
            )
            yolo_fps, yolo_time = self.measure_speed_at_threshold(
                yolo12, 'yolo', test_images, threshold
            )
            yolo_metrics['FPS'] = yolo_fps
            yolo_metrics['Speed (ms)'] = yolo_time
            yolo_metrics['Size (MB)'] = yolo12_size
            yolo_results.append(yolo_metrics)
            
            print(f"      Faster R-CNN...")
            faster_metrics = self.evaluate_at_threshold(
                faster_rcnn, 'faster_rcnn', test_images, labels_dir, threshold
            )
            faster_fps, faster_time = self.measure_speed_at_threshold(
                faster_rcnn, 'faster_rcnn', test_images, threshold
            )
            faster_metrics['FPS'] = faster_fps
            faster_metrics['Speed (ms)'] = faster_time
            faster_metrics['Size (MB)'] = faster_size
            faster_results.append(faster_metrics)
            
            print(f"      YOLOv12: mAP@0.5={yolo_metrics['mAP@0.5']:.2f}%, "
                  f"FPS={yolo_fps:.1f}, Детекций={yolo_metrics['total_detections']}, "
                  f"Ср.уверенность={yolo_metrics['avg_confidence']:.3f}")
            print(f"      Faster R-CNN: mAP@0.5={faster_metrics['mAP@0.5']:.2f}%, "
                  f"FPS={faster_fps:.1f}, Детекций={faster_metrics['total_detections']}, "
                  f"Ср.уверенность={faster_metrics['avg_confidence']:.3f}")
        
        self.detailed_results = {
            'YOLOv12': yolo_results,
            'Faster R-CNN': faster_results
        }
        
        return self.detailed_results
    
    def print_detailed_results(self):
        if not self.detailed_results:
            print("Нет результатов для отображения")
            return
        
        print("\n" + "="*110)
        print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ (усреднено по 3 прогонам)".center(110))
        print("="*110)
        
        print("\nYOLOv12:")
        print("-"*110)
        print(f"{'Порог':<8} {'mAP@0.5':>12} {'mAP@0.5:0.95':>15} {'Recall':>10} {'FPS':>8} {'Детекции':>10} {'Ср.увер.':>10}")
        print("-"*110)
        
        for r in self.detailed_results['YOLOv12']:
            print(f"{r['threshold']:<8.2f} {r['mAP@0.5']:>11.2f}% {r['mAP@0.5:0.95']:>14.2f}% "
                  f"{r['Recall@100']:>9.2f}% {r['FPS']:>7.1f} {r['total_detections']:>10} {r['avg_confidence']:>9.3f}")
        
        print("-"*110)
        
        print("\nFaster R-CNN:")
        print("-"*110)
        print(f"{'Порог':<8} {'mAP@0.5':>12} {'mAP@0.5:0.95':>15} {'Recall':>10} {'FPS':>8} {'Детекции':>10} {'Ср.увер.':>10}")
        print("-"*110)
        
        for r in self.detailed_results['Faster R-CNN']:
            print(f"{r['threshold']:<8.2f} {r['mAP@0.5']:>11.2f}% {r['mAP@0.5:0.95']:>14.2f}% "
                  f"{r['Recall@100']:>9.2f}% {r['FPS']:>7.1f} {r['total_detections']:>10} {r['avg_confidence']:>9.3f}")
        
        print("-"*110)
    
    def plot_detailed_results(self):
        if not self.detailed_results:
            print("Нет данных для визуализации")
            return
        
        thresholds = self.confidence_thresholds
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # График 1: Количество детекций
        yolo_detections = [r['total_detections'] for r in self.detailed_results['YOLOv12']]
        faster_detections = [r['total_detections'] for r in self.detailed_results['Faster R-CNN']]
        
        axes[0, 0].plot(thresholds, yolo_detections, 'o-', label='YOLOv12', color='#2E86AB', linewidth=2, markersize=8)
        axes[0, 0].plot(thresholds, faster_detections, 's-', label='Faster R-CNN', color='#A23B72', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Порог уверенности')
        axes[0, 0].set_ylabel('Количество детекций')
        axes[0, 0].set_title('Количество обнаруженных объектов')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: FPS
        yolo_fps = [r['FPS'] for r in self.detailed_results['YOLOv12']]
        faster_fps = [r['FPS'] for r in self.detailed_results['Faster R-CNN']]
        
        axes[0, 1].plot(thresholds, yolo_fps, 'o-', label='YOLOv12', color='#2E86AB', linewidth=2, markersize=8)
        axes[0, 1].plot(thresholds, faster_fps, 's-', label='Faster R-CNN', color='#A23B72', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Порог уверенности')
        axes[0, 1].set_ylabel('FPS')
        axes[0, 1].set_title('Скорость обработки')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Средняя уверенность
        yolo_avg_conf = [r['avg_confidence'] for r in self.detailed_results['YOLOv12']]
        faster_avg_conf = [r['avg_confidence'] for r in self.detailed_results['Faster R-CNN']]
        
        axes[0, 2].plot(thresholds, yolo_avg_conf, 'o-', label='YOLOv12', color='#2E86AB', linewidth=2, markersize=8)
        axes[0, 2].plot(thresholds, faster_avg_conf, 's-', label='Faster R-CNN', color='#A23B72', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Порог уверенности')
        axes[0, 2].set_ylabel('Средняя уверенность')
        axes[0, 2].set_title('Средняя уверенность модели')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # График 4: mAP@0.5
        yolo_map = [r['mAP@0.5'] for r in self.detailed_results['YOLOv12']]
        faster_map = [r['mAP@0.5'] for r in self.detailed_results['Faster R-CNN']]
        
        axes[1, 0].plot(thresholds, yolo_map, 'o-', label='YOLOv12', color='#2E86AB', linewidth=2, markersize=8)
        axes[1, 0].plot(thresholds, faster_map, 's-', label='Faster R-CNN', color='#A23B72', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Порог уверенности')
        axes[1, 0].set_ylabel('mAP@0.5 (%)')
        axes[1, 0].set_title('Точность детекции (mAP@0.5)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
        
        # График 5: mAP@0.5:0.95
        yolo_map095 = [r['mAP@0.5:0.95'] for r in self.detailed_results['YOLOv12']]
        faster_map095 = [r['mAP@0.5:0.95'] for r in self.detailed_results['Faster R-CNN']]
        
        axes[1, 1].plot(thresholds, yolo_map095, 'o-', label='YOLOv12', color='#2E86AB', linewidth=2, markersize=8)
        axes[1, 1].plot(thresholds, faster_map095, 's-', label='Faster R-CNN', color='#A23B72', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Порог уверенности')
        axes[1, 1].set_ylabel('mAP@0.5:0.95 (%)')
        axes[1, 1].set_title('Точность локализации')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        # График 6: Сравнение метрик
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Ср.уверенность']
        yolo_vals = [np.mean([r['mAP@0.5'] for r in self.detailed_results['YOLOv12']]),
                     np.mean([r['mAP@0.5:0.95'] for r in self.detailed_results['YOLOv12']]),
                     np.mean([r['avg_confidence'] for r in self.detailed_results['YOLOv12']]) * 100]
        faster_vals = [np.mean([r['mAP@0.5'] for r in self.detailed_results['Faster R-CNN']]),
                       np.mean([r['mAP@0.5:0.95'] for r in self.detailed_results['Faster R-CNN']]),
                       np.mean([r['avg_confidence'] for r in self.detailed_results['Faster R-CNN']]) * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, yolo_vals, width, label='YOLOv12', color='#2E86AB', alpha=0.8)
        axes[1, 2].bar(x + width/2, faster_vals, width, label='Faster R-CNN', color='#A23B72', alpha=0.8)
        axes[1, 2].set_ylabel('Значение (%)')
        axes[1, 2].set_title('Сводное сравнение')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Сравнение YOLOv12 и Faster R-CNN на тестовой выборке', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.detectors_dir / "test_comparison.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\nГрафик сохранен: {output_path}")
        plt.show()
    
    def save_detailed_results_csv(self):
        data = []
        
        for model_name, results in self.detailed_results.items():
            for r in results:
                data.append({
                    'Model': model_name,
                    'Confidence_Threshold': r['threshold'],
                    'mAP@0.5': r['mAP@0.5'],
                    'mAP@0.5:0.95': r['mAP@0.5:0.95'],
                    'Recall@100': r['Recall@100'],
                    'FPS': r['FPS'],
                    'Speed_ms': r['Speed (ms)'],
                    'Total_Detections': r['total_detections'],
                    'Total_GT': r['total_gt'],
                    'Avg_Confidence': r['avg_confidence']
                })
        
        df = pd.DataFrame(data)
        output_path = self.detectors_dir / "test_comparison.csv"
        df.to_csv(output_path, index=False)
        print(f"Результаты сохранены: {output_path}")


if __name__ == "__main__":
    comparator = DetailedModelComparator()
    results = comparator.run_detailed_comparison()
    
    if results:
        comparator.print_detailed_results()
        comparator.plot_detailed_results()
        comparator.save_detailed_results_csv()