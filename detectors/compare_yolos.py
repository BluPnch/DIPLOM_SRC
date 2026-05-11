# detectors/compare_yolos.py
"""Comparison of YOLOv11, YOLOv12 and YOLOv26 on test dataset"""

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataset_yaml


class YOLOComparison:
    """Comparison of YOLOv11, YOLOv12 and YOLOv26"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.detectors_dir = self.project_root / "detectors"
        
        # CORRECT PATHS based on your structure
        self.yolo26_weights = self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt"
        self.yolo12_weights = self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt"
        self.yolo11_weights = self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt"
        
        # Alternative paths for search
        self.alternative_paths = {
            'YOLOv26': [
                self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt",
                self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            ],
            'YOLOv12': [
                self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
                self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
                self.project_root / "runs" / "detect" / "train2" / "weights" / "best.pt",
            ],
            'YOLOv11': [
                self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt",
                self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            ]
        }
        
        self.results = {}
    
    def find_weights(self, model_name, primary_path):
        """Find model weights"""
        if primary_path and primary_path.exists():
            return primary_path
        
        print(f"   Searching {model_name} in alternative paths...")
        if model_name in self.alternative_paths:
            for alt_path in self.alternative_paths[model_name]:
                if alt_path.exists():
                    print(f"   Found {model_name}: {alt_path}")
                    return alt_path
        
        return None
    
    def get_test_data(self):
        """Get test images and labels"""
        data_cfg, yaml_path = dataset_yaml.load_data_cfg()
        
        if 'test' in data_cfg:
            test_dir = Path(data_cfg['test'])
        else:
            test_dir = dataset_yaml.resolve_split_images_dir(data_cfg, "val", yaml_path)
        
        test_images = list(dataset_yaml.iter_image_paths(test_dir))
        labels_dir = dataset_yaml.yolo_labels_dir(test_dir, "val" if 'test' not in data_cfg else "test")
        
        print(f"Test images: {len(test_images)}")
        return test_images, labels_dir
    
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
    
    def evaluate_accuracy(self, model, model_name, test_images, labels_dir):
        """Evaluate model accuracy"""
        print(f"\n   Evaluating {model_name} accuracy...")
        
        metric = MeanAveragePrecision()
        predictions = []
        targets = []
        
        total = len(test_images)
        for i, img_path in enumerate(test_images):
            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}/{total} images...")
            
            # Get image size
            with Image.open(img_path) as img:
                img_size = img.size
            
            # Prediction
            try:
                results = model(img_path, conf=0.25, verbose=False)
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
                else:
                    pred = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'scores': torch.zeros(0, dtype=torch.float32),
                        'labels': torch.zeros(0, dtype=torch.int64)
                    }
            except Exception as e:
                print(f"      Error predicting {img_path}: {e}")
                pred = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'scores': torch.zeros(0, dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.int64)
                }
            
            # Ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_boxes = []
            gt_labels = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0]) + 1
                                xc, yc, w, h = map(float, parts[1:5])
                                
                                x1 = (xc - w/2) * img_size[0]
                                y1 = (yc - h/2) * img_size[1]
                                x2 = (xc + w/2) * img_size[0]
                                y2 = (yc + h/2) * img_size[1]
                                
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_labels.append(class_id)
                except Exception as e:
                    print(f"      Error reading {label_path}: {e}")
            
            gt = {
                'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros(0, dtype=torch.int64)
            }
            
            predictions.append(pred)
            targets.append(gt)
        
        # Compute metrics
        try:
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
        except Exception as e:
            print(f"   Error computing metrics for {model_name}: {e}")
            return {
                'mAP@0.5': 0,
                'mAP@0.5:0.95': 0,
                'mAP@small': 0,
                'mAP@medium': 0,
                'mAP@large': 0,
                'Recall@100': 0,
            }
    
    def measure_speed(self, model, model_name, test_images, num_runs=30):
        """Measure inference speed"""
        print(f"   Measuring {model_name} speed...")
        
        if not test_images:
            return 0, 0
        
        test_img_path = test_images[0]
        
        # Warmup
        for _ in range(3):
            try:
                model(test_img_path, verbose=False)
            except:
                pass
        
        times = []
        for _ in range(num_runs):
            try:
                start = time.time()
                model(test_img_path, verbose=False)
                times.append(time.time() - start)
            except:
                continue
        
        if len(times) == 0:
            return 0, 0
        
        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        return fps, avg_time
    
    def get_model_size(self, weights_path):
        """Get model size in MB"""
        if weights_path and Path(weights_path).exists():
            return Path(weights_path).stat().st_size / (1024 * 1024)
        return 0
    
    def run_comparison(self):
        """Run full comparison"""
        print("\n" + "="*70)
        print("COMPARISON: YOLOv11 vs YOLOv12 vs YOLOv26".center(70))
        print("="*70)
        
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
        
        # Evaluate accuracy
        print("\n[2/4] Evaluating accuracy...")
        for model_name, model in models.items():
            self.results[model_name] = self.evaluate_accuracy(
                model, model_name, test_images, labels_dir
            )
        
        # Measure speed
        print("\n[3/4] Measuring speed...")
        for model_name, model in models.items():
            fps, speed_ms = self.measure_speed(model, model_name, test_images)
            self.results[model_name]['FPS'] = fps
            self.results[model_name]['Speed (ms)'] = speed_ms
        
        # Get model sizes
        print("\n[4/4] Getting model sizes...")
        for model_name in models.keys():
            if model_name == 'YOLOv11':
                weights_path = self.find_weights('YOLOv11', self.yolo11_weights)
            elif model_name == 'YOLOv12':
                weights_path = self.find_weights('YOLOv12', self.yolo12_weights)
            else:
                weights_path = self.find_weights('YOLOv26', self.yolo26_weights)
            
            size_mb = self.get_model_size(weights_path)
            self.results[model_name]['Size (MB)'] = size_mb
        
        return self.results
    
    def print_results(self):
        """Print results table"""
        if not self.results:
            print("No results to display")
            return
        
        model_names = list(self.results.keys())
        
        print("\n" + "="*90)
        print("COMPARISON RESULTS".center(90))
        print("="*90)
        
        # Accuracy table
        print("\nACCURACY:")
        print("-"*90)
        header = f"{'Metric':<20}"
        for name in model_names:
            header += f"{name:>20}"
        print(header)
        print("-"*90)
        
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Recall@100']
        for metric in metrics:
            row = f"{metric:<20}"
            for name in model_names:
                val = self.results[name].get(metric, 0)
                row += f"{val:>19.2f}%"
            print(row)
        
        # Performance table
        print("\nPERFORMANCE:")
        print("-"*90)
        
        perf_metrics = ['FPS', 'Speed (ms)', 'Size (MB)']
        for metric in perf_metrics:
            row = f"{metric:<20}"
            for name in model_names:
                val = self.results[name].get(metric, 0)
                if metric == 'FPS':
                    row += f"{val:>19.1f}"
                else:
                    row += f"{val:>19.1f}"
            print(row)
        
        print("\n" + "="*90)
    
    def plot_results(self):
        """Plot comparison charts"""
        if len(self.results) < 2:
            print("Not enough models for plotting")
            return
        
        model_names = list(self.results.keys())
        colors = ['#2E86AB', '#F18F01', '#A23B72']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Chart 1: mAP@0.5
        map_vals = [self.results[name].get('mAP@0.5', 0) for name in model_names]
        bars = axes[0, 0].bar(model_names, map_vals, color=colors[:len(model_names)], alpha=0.8)
        axes[0, 0].set_ylabel('mAP@0.5 (%)')
        axes[0, 0].set_title('Detection Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, map_vals):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                           f'{val:.1f}%', ha='center', fontweight='bold')
        
        # Chart 2: mAP@0.5:0.95
        map095_vals = [self.results[name].get('mAP@0.5:0.95', 0) for name in model_names]
        bars = axes[0, 1].bar(model_names, map095_vals, color=colors[:len(model_names)], alpha=0.8)
        axes[0, 1].set_ylabel('mAP@0.5:0.95 (%)')
        axes[0, 1].set_title('Localization Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, map095_vals):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                           f'{val:.1f}%', ha='center', fontweight='bold')
        
        # Chart 3: FPS
        fps_vals = [self.results[name].get('FPS', 0) for name in model_names]
        bars = axes[1, 0].bar(model_names, fps_vals, color=colors[:len(model_names)], alpha=0.8)
        axes[1, 0].set_ylabel('FPS')
        axes[1, 0].set_title('Inference Speed', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, fps_vals):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                           f'{val:.1f}', ha='center', fontweight='bold')
        
        # Chart 4: Model Size
        size_vals = [self.results[name].get('Size (MB)', 0) for name in model_names]
        bars = axes[1, 1].bar(model_names, size_vals, color=colors[:len(model_names)], alpha=0.8)
        axes[1, 1].set_ylabel('Size (MB)')
        axes[1, 1].set_title('Model Size', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, size_vals):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                           f'{val:.1f} MB', ha='center', fontweight='bold')
        
        plt.suptitle('YOLO Models Comparison: v11 vs v12 vs v26', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.detectors_dir / "yolo_comparison.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"\nChart saved: {output_path}")
        plt.show()
    
    def save_csv(self):
        """Save results to CSV"""
        output_path = self.detectors_dir / "yolo_comparison.csv"
        
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
        print(f"Results saved: {output_path}")
    
    def print_conclusion(self):
        """Print conclusion"""
        if len(self.results) < 2:
            print("\nNot enough data for conclusion")
            return
        
        print("\n" + "="*70)
        print("CONCLUSIONS".center(70))
        print("="*70)
        
        model_names = list(self.results.keys())
        
        # Find best in each category
        best_accuracy = max(model_names, key=lambda x: self.results[x].get('mAP@0.5', 0))
        best_speed = max(model_names, key=lambda x: self.results[x].get('FPS', 0))
        best_size = min(model_names, key=lambda x: self.results[x].get('Size (MB)', float('inf')))
        
        print(f"\n1. Best detection accuracy: {best_accuracy}")
        print(f"   mAP@0.5: {self.results[best_accuracy]['mAP@0.5']:.2f}%")
        
        print(f"\n2. Best inference speed: {best_speed}")
        print(f"   FPS: {self.results[best_speed]['FPS']:.1f}")
        
        print(f"\n3. Smallest model size: {best_size}")
        print(f"   Size: {self.results[best_size]['Size (MB)']:.1f} MB")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    comparator = YOLOComparison()
    results = comparator.run_comparison()
    
    if results:
        comparator.print_results()
        comparator.plot_results()
        comparator.save_csv()
        comparator.print_conclusion()