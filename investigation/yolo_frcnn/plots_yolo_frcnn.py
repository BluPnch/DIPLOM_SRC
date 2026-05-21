# investigation/yolo_frcnn/plots_yolo_frcnn.py
"""Построение графиков по результатам сравнения моделей с эталонной линией"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path

# Путь к файлу с результатами
RESULTS_FILE = Path(__file__).parent / "comparison_results.csv"

# Цвета и стили для моделей
MODEL_STYLES = {
    'YOLOv12': {'color': '#E69F00', 'linestyle': '-', 'linewidth': 2.5, 'label': 'YOLOv12'},
    'Faster R-CNN': {'color': '#56B4E9', 'linestyle': '--', 'linewidth': 2.5, 'label': 'Faster R-CNN'},
}

# Эталонное значение
GROUND_TRUTH_DETECTIONS = 324
GROUND_TRUTH_COLOR = 'black'
GROUND_TRUTH_LINESTYLE = '-'
GROUND_TRUTH_LINEWIDTH = 3


class ResultsPlotter:
    """Построение графиков по результатам сравнения моделей"""
    
    def __init__(self):
        self.yolo12_data = None
        self.frcnn_data = None
        self.load_data()
    
    def load_data(self):
        """Загрузка данных из CSV файлов"""
        if not RESULTS_FILE.exists():
            print(f"Файл {RESULTS_FILE} не найден")
            return
        
        df = pd.read_csv(RESULTS_FILE)
        
        if 'Confidence_Threshold' in df.columns:
            thresh_col = 'Confidence_Threshold'
        elif 'threshold' in df.columns:
            thresh_col = 'threshold'
        else:
            print(f"Доступные колонки: {df.columns.tolist()}")
            raise KeyError("Колонка с порогами не найдена")
        
        self.yolo12_data = df[df['Model'] == 'YOLOv12'].copy()
        self.yolo12_data = self.yolo12_data.sort_values(thresh_col)
        self.yolo12_data['threshold'] = self.yolo12_data[thresh_col].values
        
        self.frcnn_data = df[df['Model'] == 'Faster R-CNN'].copy()
        self.frcnn_data = self.frcnn_data.sort_values(thresh_col)
        self.frcnn_data['threshold'] = self.frcnn_data[thresh_col].values
        
        print(f"Загружены данные YOLOv12: {len(self.yolo12_data)} записей")
        print(f"Загружены данные Faster R-CNN: {len(self.frcnn_data)} записей")
    
    def smooth_curve(self, x, y, num_points=300):
        """Сглаживание кривой с помощью сплайн-интерполяции"""
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(y, 'values'):
            y = y.values
        
        # НЕ обрезаем нулевые значения, оставляем все данные
        mask = ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return x_clean, y_clean
        
        x_smooth = np.linspace(min(x_clean), max(x_clean), num_points)
        
        try:
            spline = UnivariateSpline(x_clean, y_clean, s=0.01 * len(x_clean))
            y_smooth = spline(x_smooth)
        except:
            y_smooth = np.interp(x_smooth, x_clean, y_clean)
        
        return x_smooth, y_smooth
    
    def find_intersection_point(self, x_values, y_values, target_y):
        """Нахождение точки пересечения кривой с горизонтальной линией"""
        for i in range(len(y_values) - 1):
            if (y_values[i] - target_y) * (y_values[i+1] - target_y) <= 0:
                if y_values[i+1] != y_values[i]:
                    t = (target_y - y_values[i]) / (y_values[i+1] - y_values[i])
                    x_intersect = x_values[i] + t * (x_values[i+1] - x_values[i])
                    return x_intersect
        return None
    
    def plot_detections_comparison(self):
        """График: Количество детекций vs порог с эталонной линией"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Эталонная линия
        ax.axhline(y=GROUND_TRUTH_DETECTIONS, 
                   color=GROUND_TRUTH_COLOR, 
                   linestyle=GROUND_TRUTH_LINESTYLE, 
                   linewidth=GROUND_TRUTH_LINEWIDTH,
                   label=f'Эталон ({GROUND_TRUTH_DETECTIONS} объектов)')
        
        intersections = []
        all_x_intersects = []
        
        # YOLOv12
        if self.yolo12_data is not None and len(self.yolo12_data) > 0:
            x = self.yolo12_data['threshold']
            y = self.yolo12_data['Total_Detections']
            style = MODEL_STYLES['YOLOv12']
            
            x_smooth, y_smooth = self.smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
            
            x_int = self.find_intersection_point(x_smooth, y_smooth, GROUND_TRUTH_DETECTIONS)
            if x_int is not None and 0 <= x_int <= 1:
                intersections.append(('YOLOv12', x_int))
                all_x_intersects.append(x_int)
                ax.plot(x_int, GROUND_TRUTH_DETECTIONS, 'o', color=style['color'], markersize=8, markeredgecolor='black')
        
        # Faster R-CNN
        if self.frcnn_data is not None and len(self.frcnn_data) > 0:
            x = self.frcnn_data['threshold']
            y = self.frcnn_data['Total_Detections']
            style = MODEL_STYLES['Faster R-CNN']
            
            x_smooth, y_smooth = self.smooth_curve(x, y)
            ax.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
            
            x_int = self.find_intersection_point(x_smooth, y_smooth, GROUND_TRUTH_DETECTIONS)
            if x_int is not None and 0 <= x_int <= 1:
                intersections.append(('Faster R-CNN', x_int))
                all_x_intersects.append(x_int)
                ax.plot(x_int, GROUND_TRUTH_DETECTIONS, 's', color=style['color'], markersize=8, markeredgecolor='black')
        
        ax.set_xlabel('Порог уверенности', fontsize=12)
        ax.set_ylabel('Количество обнаруженных объектов', fontsize=12)
        ax.set_title('Сравнение количества обнаруженных объектов с эталоном', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 700)
        
        # Вертикальные линии для пересечений
        for name, x_int in intersections:
            ax.axvline(x=x_int, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Настройка подписей на оси X (добавляем пороги пересечения)
        current_xticks = list(ax.get_xticks())
        for x_int in all_x_intersects:
            if x_int not in current_xticks:
                current_xticks.append(x_int)
        current_xticks = sorted(current_xticks)
        
        # Формируем подписи
        xtick_labels = []
        for val in current_xticks:
            if val in all_x_intersects:
                # Для порогов пересечения делаем подпись с названием модели
                names = [name for name, x in intersections if abs(x - val) < 0.01]
                if names:
                    xtick_labels.append(f'{val:.2f}\n({names[0]})')
                else:
                    xtick_labels.append(f'{val:.2f}')
            else:
                xtick_labels.append(f'{val:.2f}')
        
        ax.set_xticks(current_xticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        
        # Подсветка подписей для порогов пересечения
        for tick, val in zip(ax.get_xticklabels(), current_xticks):
            if val in all_x_intersects:
                tick.set_color('red')
                tick.set_fontweight('bold')
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_detections_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_fps_comparison(self):
        """График: Сравнение FPS YOLOv12 и Faster R-CNN"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is not None and len(self.yolo12_data) > 0:
            x = self.yolo12_data['threshold']
            y = self.yolo12_data['FPS']
            style = MODEL_STYLES['YOLOv12']
            x_smooth, y_smooth = self.smooth_curve(x, y)
            plt.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
        
        if self.frcnn_data is not None and len(self.frcnn_data) > 0:
            x = self.frcnn_data['threshold']
            y = self.frcnn_data['FPS']
            style = MODEL_STYLES['Faster R-CNN']
            x_smooth, y_smooth = self.smooth_curve(x, y)
            plt.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.title('Сравнение скорости обработки', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_fps_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_yolo12_all_metrics(self):
        """График: Все метрики точности YOLOv12 на одном графике"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is None or len(self.yolo12_data) == 0:
            print("Нет данных YOLOv12")
            return
        
        x = self.yolo12_data['threshold']
        
        # mAP@0.5
        y1 = self.yolo12_data['mAP@0.5']
        x_smooth, y_smooth = self.smooth_curve(x, y1)
        plt.plot(x_smooth, y_smooth, '-', linewidth=2.5, label='mAP@0.5', color='#E69F00')
        
        # mAP@0.5:0.95
        y2 = self.yolo12_data['mAP@0.5:0.95']
        x_smooth, y_smooth = self.smooth_curve(x, y2)
        plt.plot(x_smooth, y_smooth, '--', linewidth=2.5, label='mAP@0.5:0.95', color='#56B4E9')
        
        # Recall@100
        y3 = self.yolo12_data['Recall@100']
        x_smooth, y_smooth = self.smooth_curve(x, y3)
        plt.plot(x_smooth, y_smooth, '-.', linewidth=2.5, label='Recall@100', color='#009E73')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Значение (%)', fontsize=12)
        plt.title('Точность детекции YOLOv12', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_yolo12_all_metrics.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_frcnn_all_metrics(self):
        """График: Все метрики точности Faster R-CNN на одном графике"""
        plt.figure(figsize=(10, 6))
        
        if self.frcnn_data is None or len(self.frcnn_data) == 0:
            print("Нет данных Faster R-CNN")
            return
        
        x = self.frcnn_data['threshold']
        
        # mAP@0.5
        y1 = self.frcnn_data['mAP@0.5']
        x_smooth, y_smooth = self.smooth_curve(x, y1)
        plt.plot(x_smooth, y_smooth, '-', linewidth=2.5, label='mAP@0.5', color='#E69F00')
        
        # mAP@0.5:0.95
        y2 = self.frcnn_data['mAP@0.5:0.95']
        x_smooth, y_smooth = self.smooth_curve(x, y2)
        plt.plot(x_smooth, y_smooth, '--', linewidth=2.5, label='mAP@0.5:0.95', color='#56B4E9')
        
        # Recall@100
        y3 = self.frcnn_data['Recall@100']
        x_smooth, y_smooth = self.smooth_curve(x, y3)
        plt.plot(x_smooth, y_smooth, '-.', linewidth=2.5, label='Recall@100', color='#009E73')
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Значение (%)', fontsize=12)
        plt.title('Точность детекции Faster R-CNN', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_frcnn_all_metrics.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_map_comparison(self):
        """График: Сравнение mAP@0.5 YOLOv12 и Faster R-CNN"""
        plt.figure(figsize=(10, 6))
        
        if self.yolo12_data is not None and len(self.yolo12_data) > 0:
            x = self.yolo12_data['threshold']
            y = self.yolo12_data['mAP@0.5']
            style = MODEL_STYLES['YOLOv12']
            x_smooth, y_smooth = self.smooth_curve(x, y)
            plt.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
        
        if self.frcnn_data is not None and len(self.frcnn_data) > 0:
            x = self.frcnn_data['threshold']
            y = self.frcnn_data['mAP@0.5']
            style = MODEL_STYLES['Faster R-CNN']
            x_smooth, y_smooth = self.smooth_curve(x, y)
            plt.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('mAP@0.5 (%)', fontsize=12)
        plt.title('Сравнение точности детекции', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_map_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_detections_comparison()
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