# detectors/plot_results.py
"""Построение графиков по результатам сравнения моделей с интерполяцией"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pathlib import Path

# Пути к файлам с результатами
REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTORS_DIR = REPO_ROOT / "detectors"

# Файлы результатов
YOLO_COMPARISON_FILE = DETECTORS_DIR / "yolo_detailed_comparison.csv"
FRCNN_COMPARISON_FILE = DETECTORS_DIR / "test_comparison.csv"

# Стили линий для чёрно-белой печати
LINESTYLES = {
    'solid': '-',           # сплошная
    'dashed': '--',         # пунктир (короткий)
    'dashdot': '-.',        # штрихпунктир
    'dotted': ':',          # точки
}

class ResultsPlotter:
    """Построение графиков по результатам сравнения моделей"""
    
    def __init__(self):
        self.yolo12_data = None
        self.frcnn_data = None
        self.load_data()
    
    def load_data(self):
        """Загрузка данных из CSV файлов"""
        if YOLO_COMPARISON_FILE.exists():
            df = pd.read_csv(YOLO_COMPARISON_FILE)
            self.yolo12_data = df[df['Model'] == 'YOLOv12'].copy()
            self.yolo12_data = self.yolo12_data.sort_values('Confidence_Threshold')
            print(f"Загружены данные YOLOv12: {len(self.yolo12_data)} записей")
        else:
            print(f"Файл {YOLO_COMPARISON_FILE} не найден")
            self.yolo12_data = None
        
        if FRCNN_COMPARISON_FILE.exists():
            df = pd.read_csv(FRCNN_COMPARISON_FILE)
            self.frcnn_data = df[df['Model'] == 'Faster R-CNN'].copy()
            self.frcnn_data = self.frcnn_data.sort_values('Confidence_Threshold')
            print(f"Загружены данные Faster R-CNN: {len(self.frcnn_data)} записей")
        else:
            print(f"Файл {FRCNN_COMPARISON_FILE} не найден")
            self.frcnn_data = None
    
    def interpolate_curve(self, x, y, num_points=100):
        """Сглаживание кривой между точками"""
        mask = ~np.isnan(y)
        x_clean = np.array(x)[mask]
        y_clean = np.array(y)[mask]
        
        if len(x_clean) < 3:
            return x_clean, y_clean
        
        x_smooth = np.linspace(min(x_clean), max(x_clean), num_points)
        
        try:
            spline = make_interp_spline(x_clean, y_clean, k=3, bc_type='natural')
            y_smooth = spline(x_smooth)
        except:
            from scipy.interpolate import interp1d
            f = interp1d(x_clean, y_clean, kind='cubic', fill_value='extrapolate')
            y_smooth = f(x_smooth)
        
        return x_smooth, y_smooth
    
    def plot_fps_comparison(self):
        """График 1: Сравнение FPS YOLOv12 и Faster R-CNN"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is not None:
            x = self.yolo12_data['Confidence_Threshold'].values
            y = self.yolo12_data['FPS'].values
            x_smooth, y_smooth = self.interpolate_curve(x, y)
            plt.plot(x_smooth, y_smooth, LINESTYLES['solid'], 
                    linewidth=2.5, label='YOLOv12', color='black')
        
        if self.frcnn_data is not None:
            x = self.frcnn_data['Confidence_Threshold'].values
            y = self.frcnn_data['FPS'].values
            x_smooth, y_smooth = self.interpolate_curve(x, y)
            plt.plot(x_smooth, y_smooth, LINESTYLES['dashed'], 
                    linewidth=2.5, label='Faster R-CNN', color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('Сравнение скорости обработки', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_fps_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_fps_comparison.png")
    
    def plot_yolo12_all_metrics(self):
        """График 2: Все метрики точности YOLOv12 на одном графике"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is None:
            print("Нет данных YOLOv12")
            return
        
        x = self.yolo12_data['Confidence_Threshold'].values
        
        # mAP@0.5
        y1 = self.yolo12_data['mAP@0.5'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y1)
        plt.plot(x_smooth, y_smooth, LINESTYLES['solid'], 
                linewidth=2.5, label='mAP@0.5', color='black')
        
        # mAP@0.5:0.95
        y2 = self.yolo12_data['mAP@0.5:0.95'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y2)
        plt.plot(x_smooth, y_smooth, LINESTYLES['dashed'], 
                linewidth=2.5, label='mAP@0.5:0.95', color='black')
        
        # Recall@100
        y3 = self.yolo12_data['Recall@100'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y3)
        plt.plot(x_smooth, y_smooth, LINESTYLES['dashdot'], 
                linewidth=2.5, label='Recall@100', color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Значение (%)', fontsize=12)
        plt.title('Точность детекции YOLOv12', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_yolo12_all_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_yolo12_all_metrics.png")
    
    def plot_frcnn_all_metrics(self):
        """График 3: Все метрики точности Faster R-CNN на одном графике"""
        plt.figure(figsize=(10, 6))
        
        if self.frcnn_data is None:
            print("Нет данных Faster R-CNN")
            return
        
        x = self.frcnn_data['Confidence_Threshold'].values
        
        # mAP@0.5
        y1 = self.frcnn_data['mAP@0.5'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y1)
        plt.plot(x_smooth, y_smooth, LINESTYLES['solid'], 
                linewidth=2.5, label='mAP@0.5', color='black')
        
        # mAP@0.5:0.95
        y2 = self.frcnn_data['mAP@0.5:0.95'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y2)
        plt.plot(x_smooth, y_smooth, LINESTYLES['dashed'], 
                linewidth=2.5, label='mAP@0.5:0.95', color='black')
        
        # Recall@100
        y3 = self.frcnn_data['Recall@100'].values
        x_smooth, y_smooth = self.interpolate_curve(x, y3)
        plt.plot(x_smooth, y_smooth, LINESTYLES['dashdot'], 
                linewidth=2.5, label='Recall@100', color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Значение (%)', fontsize=12)
        plt.title('Точность детекции Faster R-CNN', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_frcnn_all_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_frcnn_all_metrics.png")
    
    def plot_map_comparison(self):
        """График 4: Сравнение mAP@0.5 YOLOv12 и Faster R-CNN"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is not None:
            x = self.yolo12_data['Confidence_Threshold'].values
            y = self.yolo12_data['mAP@0.5'].values
            x_smooth, y_smooth = self.interpolate_curve(x, y)
            plt.plot(x_smooth, y_smooth, LINESTYLES['solid'], 
                    linewidth=2.5, label='YOLOv12', color='black')
        
        if self.frcnn_data is not None:
            x = self.frcnn_data['Confidence_Threshold'].values
            y = self.frcnn_data['mAP@0.5'].values
            x_smooth, y_smooth = self.interpolate_curve(x, y)
            plt.plot(x_smooth, y_smooth, LINESTYLES['dashed'], 
                    linewidth=2.5, label='Faster R-CNN', color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('mAP@0.5 (%)', fontsize=12)
        plt.title('Сравнение точности детекции', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_map_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_map_comparison.png")
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_fps_comparison()
        self.plot_yolo12_all_metrics()
        self.plot_frcnn_all_metrics()
        self.plot_map_comparison()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


if __name__ == "__main__":
    plotter = ResultsPlotter()
    plotter.plot_all()