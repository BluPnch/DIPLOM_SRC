# detectors/plot_yolo_comparison.py
"""Построение графиков для сравнения YOLOv11, YOLOv12 и YOLOv26"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTORS_DIR = REPO_ROOT / "detectors"

# Файл с результатами
YOLO_FILE = DETECTORS_DIR / "yolo_detailed_comparison.csv"

# Стили линий для чёрно-белой печати
LINESTYLES = {
    'solid': '-',
    'dashed': '--',
    'dashdot': '-.',
}

class YOLOPlotter:
    def __init__(self):
        self.data = None
        self.load_data()
    
    def load_data(self):
        if YOLO_FILE.exists():
            df = pd.read_csv(YOLO_FILE)
            self.data = {
                'YOLOv11': df[df['Model'] == 'YOLOv11'].sort_values('Confidence_Threshold'),
                'YOLOv12': df[df['Model'] == 'YOLOv12'].sort_values('Confidence_Threshold'),
                'YOLOv26': df[df['Model'] == 'YOLOv26'].sort_values('Confidence_Threshold')
            }
            print(f"Загружены данные: YOLOv11={len(self.data['YOLOv11'])}, YOLOv12={len(self.data['YOLOv12'])}, YOLOv26={len(self.data['YOLOv26'])}")
        else:
            print(f"Файл {YOLO_FILE} не найден")
    
    def interpolate_curve(self, x, y, num_points=100):
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
        """График 1: Сравнение FPS трёх моделей"""
        plt.figure(figsize=(10, 6))
        
        models = ['YOLOv11', 'YOLOv12', 'YOLOv26']
        styles = [LINESTYLES['solid'], LINESTYLES['dashed'], LINESTYLES['dashdot']]
        
        for model, style in zip(models, styles):
            if self.data[model] is not None and len(self.data[model]) > 0:
                x = self.data[model]['Confidence_Threshold'].values
                y = self.data[model]['FPS'].values
                x_smooth, y_smooth = self.interpolate_curve(x, y)
                plt.plot(x_smooth, y_smooth, style, linewidth=2.5, label=model, color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('Сравнение скорости обработки YOLO моделей', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_yolo_fps_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_yolo_fps_comparison.png")
    
    def plot_map_comparison(self):
        """График 2: Сравнение mAP@0.5 трёх моделей"""
        plt.figure(figsize=(10, 6))
        
        models = ['YOLOv11', 'YOLOv12', 'YOLOv26']
        styles = [LINESTYLES['solid'], LINESTYLES['dashed'], LINESTYLES['dashdot']]
        
        for model, style in zip(models, styles):
            if self.data[model] is not None and len(self.data[model]) > 0:
                x = self.data[model]['Confidence_Threshold'].values
                y = self.data[model]['mAP@0.5'].values
                x_smooth, y_smooth = self.interpolate_curve(x, y)
                plt.plot(x_smooth, y_smooth, style, linewidth=2.5, label=model, color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('mAP@0.5 (%)', fontsize=12)
        plt.title('Сравнение точности детекции YOLO моделей', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_yolo_map_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_yolo_map_comparison.png")
    
    def plot_recall_comparison(self):
        """График 3: Сравнение Recall трёх моделей"""
        plt.figure(figsize=(10, 6))
        
        models = ['YOLOv11', 'YOLOv12', 'YOLOv26']
        styles = [LINESTYLES['solid'], LINESTYLES['dashed'], LINESTYLES['dashdot']]
        
        for model, style in zip(models, styles):
            if self.data[model] is not None and len(self.data[model]) > 0:
                x = self.data[model]['Confidence_Threshold'].values
                y = self.data[model]['Recall@100'].values
                x_smooth, y_smooth = self.interpolate_curve(x, y)
                plt.plot(x_smooth, y_smooth, style, linewidth=2.5, label=model, color='black')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Recall@100 (%)', fontsize=12)
        plt.title('Сравнение полноты обнаружения YOLO моделей', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(DETECTORS_DIR / 'plot_yolo_recall_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_yolo_recall_comparison.png")
    
    def plot_all(self):
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ YOLO МОДЕЛЕЙ".center(60))
        print("="*60 + "\n")
        
        self.plot_fps_comparison()
        self.plot_map_comparison()
        self.plot_recall_comparison()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


if __name__ == "__main__":
    plotter = YOLOPlotter()
    plotter.plot_all()