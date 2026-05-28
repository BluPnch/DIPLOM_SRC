# detectors/compare_yolos_detailed.py
"""Детальное сравнение YOLOv11, YOLOv12 и YOLOv26 с вариацией порога уверенности"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchmetrics.detection import MeanAveragePrecision

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.paths import DATASET_DIR, DETECTORS_DIR, REPO_ROOT


class YOLODetailedComparison:
    """Детальное сравнение YOLOv11, YOLOv12 и YOLOv26 с вариацией порога уверенности"""
    
    def __init__(self):
        self.project_root = REPO_ROOT
        self.detectors_dir = DETECTORS_DIR
        
        # Пути к моделям
        self.yolo26_weights = self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt"
        self.yolo12_weights = self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt"
        self.yolo11_weights = self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt"
        
        # Альтернативные пути для поиска
        self.alternative_paths = {
            'YOLOv26': [
                self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt",
            ],
            'YOLOv12': [
                self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
            ],
            'YOLOv11': [
                self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt",
            ]
        }
        
        # Мелкая сетка порогов (0.01, 0.03, ..., 0.99)
        self.confidence_thresholds = np.arange(0.01, 1.0, 0.02)
        
        self.results = {}
        self.detailed_results = {}
    
    def find_weights(self, model_name, primary_path):
        """Find model weights"""
        if primary_path and primary_path.exists():
            return primary_path
        
        if model_name in self.alternative_paths:
            for alt_path in self.alternative_paths[model_name]:
                if alt_path.exists():
                    return alt_path
        
        return None
    
    def get_test_data(self):
        """Get test images and labels"""
        test_images_dir = DATASET_DIR / "images" / "test"
        test_labels_dir = DATASET_DIR / "labels" / "test"
        
        if not test_images_dir.exists():
            print(f"Test images folder not found: {test_images_dir}")
            return [], None
        
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.png"))
        test_images = sorted(test_images)
        
        print(f"Test images: {len(test_images)}")
        print(f"Test images path: {test_images_dir}")
        print(f"Labels path: {test_labels_dir}")
        
        return test_images, test_labels_dir
    
    def load_model(self, model_name, weights_path):
        """Load YOLO model"""
        weights_path = self.find_weights(model_name, weights_path)
        if weights_path is None:
            print(f"   {model_name} not found, skipping...")
            return None
        
        try:
            model = YOLO(str(weights_path))
            print(f"   {model_name} loaded successfully")
            return model
        except Exception as e:
            print(f"   Error loading {model_name}: {e}")
            return None
    
    def predict_yolo(self, model, image_path, conf_threshold):
        """Prediction with YOLO at given threshold"""
        results = model(image_path, conf=conf_threshold, verbose=False)
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int) + 1
        else:
            boxes = np.array([])
            scores = np.array([])
            labels = np.array([])
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32) if len(scores) > 0 else torch.zeros(0, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(0, dtype=torch.int64),
            'num_detections': len(boxes),
            'confidence_scores': scores
        }
    
    def load_ground_truth_yolo(self, label_path, image_size):
        """Load ground truth from YOLO format"""
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
    
    def evaluate_at_threshold(self, model, model_name, test_images, labels_dir, conf_threshold):
        """Evaluate model at given confidence threshold (averaged over 3 runs)"""
        
        num_runs = 3
        all_results = []
        all_confidences = []
        
        for run in range(num_runs):
            metric = MeanAveragePrecision()
            predictions = []
            targets = []
            total_detections = 0
            total_gt = 0
            run_confidences = []
            
            for img_path in test_images:
                with Image.open(img_path) as img:
                    img_size = img.size
                
                pred = self.predict_yolo(model, img_path, conf_threshold)
                
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
        
        # Average results over 3 runs
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
    
    def measure_speed(self, model, model_name, test_images, conf_threshold, num_runs=30):
        """Measure inference speed at given threshold"""
        if not test_images:
            return 0, 0
        
        test_img_path = test_images[0]
        
        # Warmup
        for _ in range(5):
            model(test_img_path, conf=conf_threshold, verbose=False)
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(test_img_path, conf=conf_threshold, verbose=False)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        return fps, avg_time
    
    def get_model_size(self, weights_path):
        """Get model size in MB"""
        if weights_path and Path(weights_path).exists():
            return Path(weights_path).stat().st_size / (1024 * 1024)
        return 0
    
    def run_detailed_comparison(self):
        """Run detailed comparison with threshold variation"""
        print("\n" + "="*80)
        print("DETAILED COMPARISON: YOLOv11 vs YOLOv12 vs YOLOv26".center(80))
        print(f"КОЛИЧЕСТВО ПОРОГОВ: {len(self.confidence_thresholds)}".center(80))
        print("="*80)
        
        # Get test data
        test_images, labels_dir = self.get_test_data()
        
        if len(test_images) == 0:
            print("No test images found!")
            return None
        
        # Load models
        print("\n[1/4] Loading models...")
        models = {}
        
        yolo11 = self.load_model('YOLOv11', self.yolo11_weights)
        if yolo11 is not None:
            models['YOLOv11'] = yolo11
        
        yolo12 = self.load_model('YOLOv12', self.yolo12_weights)
        if yolo12 is not None:
            models['YOLOv12'] = yolo12
        
        yolo26 = self.load_model('YOLOv26', self.yolo26_weights)
        if yolo26 is not None:
            models['YOLOv26'] = yolo26
        
        if len(models) == 0:
            print("No models loaded!")
            return None
        
        # Get model sizes (independent of threshold)
        model_sizes = {}
        for model_name in models.keys():
            if model_name == 'YOLOv11':
                weights_path = self.find_weights('YOLOv11', self.yolo11_weights)
            elif model_name == 'YOLOv12':
                weights_path = self.find_weights('YOLOv12', self.yolo12_weights)
            else:
                weights_path = self.find_weights('YOLOv26', self.yolo26_weights)
            model_sizes[model_name] = self.get_model_size(weights_path)
        
        # Collect results for different thresholds
        print("\n[2/4] Evaluating accuracy at different thresholds...")
        print("-" * 80)
        
        self.detailed_results = {model_name: [] for model_name in models.keys()}
        
        for threshold in tqdm(self.confidence_thresholds, desc="Обработка порогов"):
            for model_name, model in models.items():
                metrics = self.evaluate_at_threshold(
                    model, model_name, test_images, labels_dir, threshold
                )
                fps, speed_ms = self.measure_speed(model, model_name, test_images, threshold)
                
                metrics['FPS'] = fps
                metrics['Speed (ms)'] = speed_ms
                metrics['Size (MB)'] = model_sizes[model_name]
                
                self.detailed_results[model_name].append(metrics)
        
        return self.detailed_results
    
    def print_detailed_results(self):
        """Print detailed results table"""
        if not self.detailed_results:
            print("No results to display")
            return
        
        model_names = list(self.detailed_results.keys())
        
        print("\n" + "="*120)
        print("DETAILED RESULTS AT DIFFERENT THRESHOLDS".center(120))
        print("="*120)
        
        for model_name in model_names:
            print(f"\n{model_name}:")
            print("-"*110)
            print(f"{'Threshold':<10} {'mAP@0.5':>12} {'mAP@0.5:0.95':>15} {'Recall':>10} {'FPS':>8} {'Detections':>12} {'AvgConf':>10}")
            print("-"*110)
            
            for r in self.detailed_results[model_name]:
                print(f"{r['threshold']:<10.3f} {r['mAP@0.5']:>11.2f}% {r['mAP@0.5:0.95']:>14.2f}% "
                      f"{r['Recall@100']:>9.2f}% {r['FPS']:>7.1f} {r['total_detections']:>12} {r['avg_confidence']:>9.3f}")
            
            print("-"*110)
    
    def plot_detailed_results(self):
        """Plot dependency of metrics on confidence threshold"""
        if not self.detailed_results:
            print("No data for visualization")
            return
        
        thresholds = self.confidence_thresholds
        model_names = list(self.detailed_results.keys())
        colors = {'YOLOv11': '#2E86AB', 'YOLOv12': '#F18F01', 'YOLOv26': '#A23B72'}
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Plot 1: Detections vs threshold
        for model_name in model_names:
            detections = [r['total_detections'] for r in self.detailed_results[model_name]]
            axes[0, 0].plot(thresholds, detections, 'o-', label=model_name, 
                           color=colors[model_name], linewidth=1.5, markersize=3)
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].set_title('Detection Count vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, 1)
        
        # Plot 2: FPS vs threshold
        for model_name in model_names:
            fps = [r['FPS'] for r in self.detailed_results[model_name]]
            axes[0, 1].plot(thresholds, fps, 'o-', label=model_name, 
                           color=colors[model_name], linewidth=1.5, markersize=3)
        axes[0, 1].set_xlabel('Confidence Threshold')
        axes[0, 1].set_ylabel('FPS')
        axes[0, 1].set_title('Inference Speed vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, 1)
        
        # Plot 3: Average confidence vs threshold
        for model_name in model_names:
            avg_conf = [r['avg_confidence'] for r in self.detailed_results[model_name]]
            axes[0, 2].plot(thresholds, avg_conf, 'o-', label=model_name, 
                           color=colors[model_name], linewidth=1.5, markersize=3)
        axes[0, 2].set_xlabel('Confidence Threshold')
        axes[0, 2].set_ylabel('Average Confidence')
        axes[0, 2].set_title('Average Confidence vs Threshold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].set_xlim(0, 1)
        
        # Plot 4: mAP@0.5 vs threshold
        for model_name in model_names:
            map_vals = [r['mAP@0.5'] for r in self.detailed_results[model_name]]
            axes[1, 0].plot(thresholds, map_vals, 'o-', label=model_name, 
                           color=colors[model_name], linewidth=1.5, markersize=3)
        axes[1, 0].set_xlabel('Confidence Threshold')
        axes[1, 0].set_ylabel('mAP@0.5 (%)')
        axes[1, 0].set_title('Detection Accuracy vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].set_xlim(0, 1)
        
        # Plot 5: mAP@0.5:0.95 vs threshold
        for model_name in model_names:
            map095_vals = [r['mAP@0.5:0.95'] for r in self.detailed_results[model_name]]
            axes[1, 1].plot(thresholds, map095_vals, 'o-', label=model_name, 
                           color=colors[model_name], linewidth=1.5, markersize=3)
        axes[1, 1].set_xlabel('Confidence Threshold')
        axes[1, 1].set_ylabel('mAP@0.5:0.95 (%)')
        axes[1, 1].set_title('Localization Accuracy vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_xlim(0, 1)
        
        # Plot 6: Summary comparison
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'AvgConf']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            values = [
                np.mean([r['mAP@0.5'] for r in self.detailed_results[model_name]]),
                np.mean([r['mAP@0.5:0.95'] for r in self.detailed_results[model_name]]),
                np.mean([r['avg_confidence'] for r in self.detailed_results[model_name]]) * 100
            ]
            offset = (i - len(model_names)/2 + 0.5) * width
            axes[1, 2].bar(x + offset, values, width, label=model_name, 
                          color=colors[model_name], alpha=0.8)
        
        axes[1, 2].set_ylabel('Value (%)')
        axes[1, 2].set_title('Summary Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('YOLO Models Detailed Comparison: v11 vs v12 vs v26', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.detectors_dir / "yolo_detailed_comparison.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\nChart saved: {output_path}")
        plt.show()
    
    def save_detailed_results_csv(self):
        """Save detailed results to CSV"""
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
        output_path = self.detectors_dir / "yolo_detailed_comparison.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved: {output_path}")
    
    def print_optimal_thresholds(self):
        """Determine optimal threshold for each model"""
        print("\n" + "="*70)
        print("OPTIMAL CONFIDENCE THRESHOLD".center(70))
        print("="*70)
        
        for model_name, results in self.detailed_results.items():
            # Find optimal threshold (maximizing mAP@0.5 * Recall)
            best = max(results, key=lambda x: x['mAP@0.5'] * x['Recall@100'])
            print(f"\nRecommended threshold for {model_name}: {best['threshold']:.3f}")
            print(f"   mAP@0.5 = {best['mAP@0.5']:.2f}%")
            print(f"   Recall@100 = {best['Recall@100']:.2f}%")
            print(f"   FPS = {best['FPS']:.1f}")
            print(f"   Avg Confidence = {best['avg_confidence']:.3f}")
        
        print("\n" + "="*70)
        print("RECOMMENDATION:")
        print("   YOLOv12 shows the best overall performance")
        print("="*70)


if __name__ == "__main__":
    comparator = YOLODetailedComparison()
    results = comparator.run_detailed_comparison()
    
    if results:
        comparator.print_detailed_results()
        comparator.plot_detailed_results()
        comparator.save_detailed_results_csv()
        comparator.print_optimal_thresholds()