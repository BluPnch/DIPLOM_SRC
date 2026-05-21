# investigation/yolo_frcnn/compare_yolo12_vs_rcnn.py
"""Детальное сравнение YOLOv12 и Faster R-CNN с мелкой сеткой порогов"""

import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class FastDetailedComparator:
    """Детальное сравнение YOLOv12 и Faster R-CNN с мелкой сеткой порогов"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        
        # Пути к моделям
        self.yolo12_path = Path("C:/sem8/VKR/DIPLOM_SRC/detectors/yolo12/runs/train/weights/best.pt")
        self.frcnn_path = Path("C:/sem8/VKR/DIPLOM_SRC/detectors/faster_rcnn/faster_rcnn_best.pth")
        
        # Мелкая сетка порогов: 0.01, 0.03, 0.05, ..., 0.99
        self.thresholds = np.arange(0.01, 1.0, 0.02)
        print(f"Количество порогов: {len(self.thresholds)}")
        
        # Кеширование
        self._test_images = None
        self._labels_dir = None
        self._image_sizes = {}
        
        self.results = {'YOLOv12': [], 'Faster R-CNN': []}
    
    def get_test_data(self):
        """Кешированное получение тестовых данных"""
        if self._test_images is not None:
            return self._test_images, self._labels_dir
        
        test_images_dir = Path("C:/sem8/VKR/DIPLOM_SRC/dataset/images/test")
        self._labels_dir = Path("C:/sem8/VKR/DIPLOM_SRC/dataset/labels/test")
        
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {test_images_dir}")
        
        self._test_images = []
        for img_path in sorted(test_images_dir.glob("*.jpg")):
            self._test_images.append(img_path)
            with Image.open(img_path) as img:
                self._image_sizes[img_path] = img.size
        
        print(f"Тестовых изображений: {len(self._test_images)}")
        return self._test_images, self._labels_dir
    
    def load_models(self):
        """Загрузка моделей"""
        print("\n[1/3] Загрузка моделей...")
        
        if not self.yolo12_path.exists():
            raise FileNotFoundError(f"YOLOv12 не найдена: {self.yolo12_path}")
        yolo12 = YOLO(str(self.yolo12_path))
        print("   YOLOv12 загружена")
        
        if not self.frcnn_path.exists():
            raise FileNotFoundError(f"Faster R-CNN не найдена: {self.frcnn_path}")
        
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
        faster_rcnn.load_state_dict(torch.load(str(self.frcnn_path), map_location=device))
        faster_rcnn.eval()
        faster_rcnn.to(device)
        print("   Faster R-CNN загружена")
        
        return yolo12, faster_rcnn
    
    def load_ground_truth(self, label_path, image_size):
        """Загрузка ground truth из YOLO формата"""
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
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'num_gt': len(boxes)
        }
    
    def measure_speed_at_threshold(self, model, model_type, test_img_path, conf_threshold, num_runs=10):
        """Измерение скорости при конкретном пороге"""
        # Прогрев
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
            start = time.perf_counter()
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
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        return fps, avg_time
    
    def evaluate_at_threshold(self, model, model_type, test_images, conf_threshold):
        """Оценка для одного порога"""
        metric = MeanAveragePrecision()
        predictions = []
        targets = []
        total_detections = 0
        total_gt = 0
        all_scores = []
        
        for img_path in test_images:
            img_size = self._image_sizes[img_path]
            
            if model_type == 'yolo':
                results = model(img_path, conf=conf_threshold, verbose=False)
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    labels_pred = result.boxes.cls.cpu().numpy().astype(int) + 1
                    pred = {
                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                        'scores': torch.tensor(scores, dtype=torch.float32),
                        'labels': torch.tensor(labels_pred, dtype=torch.int64)
                    }
                    total_detections += len(boxes)
                    all_scores.extend(scores)
                else:
                    pred = {'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.int64)}
            else:
                image = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img_size
                image = image.resize((640, 640), Image.Resampling.BILINEAR)
                img_tensor = transforms.ToTensor()(image).unsqueeze(0)
                device = next(model.parameters()).device
                img_tensor = img_tensor.to(device)
                
                with torch.no_grad():
                    predictions_model = model(img_tensor)
                pred_raw = predictions_model[0]
                keep = pred_raw['scores'] > conf_threshold
                boxes = pred_raw['boxes'][keep].cpu().numpy()
                scores = pred_raw['scores'][keep].cpu().numpy()
                labels_pred = pred_raw['labels'][keep].cpu().numpy()
                
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= orig_w / 640
                    boxes[:, [1, 3]] *= orig_h / 640
                
                pred = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
                    'scores': torch.tensor(scores, dtype=torch.float32) if len(scores) > 0 else torch.zeros(0),
                    'labels': torch.tensor(labels_pred, dtype=torch.int64) if len(labels_pred) > 0 else torch.zeros(0, dtype=torch.int64)
                }
                total_detections += len(boxes)
                all_scores.extend(scores)
            
            label_path = self._labels_dir / f"{img_path.stem}.txt"
            gt = self.load_ground_truth(label_path, img_size)
            total_gt += gt['num_gt']
            
            predictions.append(pred)
            targets.append(gt)
        
        metric.update(predictions, targets)
        results = metric.compute()
        
        return {
            'threshold': conf_threshold,
            'mAP@0.5': results['map_50'].item() * 100,
            'mAP@0.5:0.95': results['map'].item() * 100,
            'Recall@100': results.get('mar_100', torch.tensor(0)).item() * 100,
            'total_detections': total_detections,
            'total_gt': total_gt,
            'avg_confidence': np.mean(all_scores) if all_scores else 0
        }
    
    def get_model_size(self, model_path):
        """Размер модели в MB"""
        return model_path.stat().st_size / (1024 * 1024)
    
    def run_comparison(self):
        """Запуск полного сравнения"""
        print("\n" + "="*70)
        print("СРАВНЕНИЕ YOLOv12 vs FASTER R-CNN".center(70))
        print(f"КОЛИЧЕСТВО ПОРОГОВ: {len(self.thresholds)}".center(70))
        print("="*70)
        
        test_images, _ = self.get_test_data()
        if not test_images:
            print("Нет тестовых изображений!")
            return None
        
        yolo12, faster_rcnn = self.load_models()
        
        # Размеры моделей
        yolo_size = self.get_model_size(self.yolo12_path)
        faster_size = self.get_model_size(self.frcnn_path)
        
        print("\n[2/3] Оценка точности и скорости...")
        
        # Берём одно изображение для замера скорости
        speed_test_img = test_images[0]
        
        for threshold in tqdm(self.thresholds, desc="Обработка порогов"):
            # YOLOv12
            yolo_res = self.evaluate_at_threshold(yolo12, 'yolo', test_images, threshold)
            yolo_fps, yolo_time = self.measure_speed_at_threshold(yolo12, 'yolo', speed_test_img, threshold)
            yolo_res['FPS'] = yolo_fps
            yolo_res['Speed_ms'] = yolo_time
            yolo_res['Size_MB'] = yolo_size
            self.results['YOLOv12'].append(yolo_res)
            
            # Faster R-CNN
            faster_res = self.evaluate_at_threshold(faster_rcnn, 'faster_rcnn', test_images, threshold)
            faster_fps, faster_time = self.measure_speed_at_threshold(faster_rcnn, 'faster_rcnn', speed_test_img, threshold)
            faster_res['FPS'] = faster_fps
            faster_res['Speed_ms'] = faster_time
            faster_res['Size_MB'] = faster_size
            self.results['Faster R-CNN'].append(faster_res)
        
        return self.results
    
    def save_csv(self, output_path="comparison_results.csv"):
        """Сохранение результатов в CSV"""
        data = []
        for model_name, results in self.results.items():
            for r in results:
                data.append({
                    'Model': model_name,
                    'Confidence_Threshold': r['threshold'],
                    'mAP@0.5': r['mAP@0.5'],
                    'mAP@0.5:0.95': r['mAP@0.5:0.95'],
                    'Recall@100': r['Recall@100'],
                    'FPS': r['FPS'],
                    'Speed_ms': r['Speed_ms'],
                    'Total_Detections': r['total_detections'],
                    'Total_GT': r['total_gt'],
                    'Avg_Confidence': r['avg_confidence']
                })
        
        df = pd.DataFrame(data)
        output_file = Path(__file__).parent / output_path
        df.to_csv(output_file, index=False)
        print(f"Результаты сохранены в {output_file}")
    
    def plot_results(self):
        """Построение графиков"""
        if not self.results['YOLOv12']:
            return
        
        df_yolo = pd.DataFrame(self.results['YOLOv12'])
        df_frcnn = pd.DataFrame(self.results['Faster R-CNN'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        thresholds = df_yolo['threshold'].values
        
        # mAP@0.5
        axes[0, 0].plot(thresholds, df_yolo['mAP@0.5'].values, '-', 
                       linewidth=2, label='YOLOv12', color='#E69F00')
        axes[0, 0].plot(thresholds, df_frcnn['mAP@0.5'].values, '--', 
                       linewidth=2, label='Faster R-CNN', color='#56B4E9')
        axes[0, 0].set_xlabel('Порог уверенности')
        axes[0, 0].set_ylabel('mAP@0.5 (%)')
        axes[0, 0].set_title('Точность детекции')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # Recall
        axes[0, 1].plot(thresholds, df_yolo['Recall@100'].values, '-', 
                       linewidth=2, label='YOLOv12', color='#E69F00')
        axes[0, 1].plot(thresholds, df_frcnn['Recall@100'].values, '--', 
                       linewidth=2, label='Faster R-CNN', color='#56B4E9')
        axes[0, 1].set_xlabel('Порог уверенности')
        axes[0, 1].set_ylabel('Recall@100 (%)')
        axes[0, 1].set_title('Полнота обнаружения')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # FPS
        axes[1, 0].plot(thresholds, df_yolo['FPS'].values, '-', 
                       linewidth=2, label='YOLOv12', color='#E69F00')
        axes[1, 0].plot(thresholds, df_frcnn['FPS'].values, '--', 
                       linewidth=2, label='Faster R-CNN', color='#56B4E9')
        axes[1, 0].set_xlabel('Порог уверенности')
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].set_title('Скорость обработки')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Количество детекций
        axes[1, 1].plot(thresholds, df_yolo['Total_Detections'].values, '-', 
                       linewidth=2, label='YOLOv12', color='#E69F00')
        axes[1, 1].plot(thresholds, df_frcnn['Total_Detections'].values, '--', 
                       linewidth=2, label='Faster R-CNN', color='#56B4E9')
        axes[1, 1].axhline(y=324, color='black', linestyle='-', linewidth=2, label='Эталон (324 объекта)')
        axes[1, 1].set_xlabel('Порог уверенности')
        axes[1, 1].set_ylabel('Количество детекций')
        axes[1, 1].set_title('Обнаруженные объекты')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Сравнение YOLOv12 и Faster R-CNN', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = Path(__file__).parent / "comparison_plots.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Графики сохранены в {output_path}")


if __name__ == "__main__":
    comparator = FastDetailedComparator()
    results = comparator.run_comparison()
    
    if results:
        comparator.save_csv()
        comparator.plot_results()
        print("\n✅ Исследование завершено!")