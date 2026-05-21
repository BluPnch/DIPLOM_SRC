# investigation/yolos/plot_yolo_comparison.py
"""Построение графиков сравнения YOLOv11, YOLOv12 и YOLOv26 с эталонной линией"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from pathlib import Path

# Путь к файлу с результатами
RESULTS_FILE = Path(__file__).parent / "yolo_detailed_comparison.csv"

# Стили для разных моделей
MODEL_STYLES = {
    'YOLOv11': {'color': '#2E86AB', 'linestyle': '--', 'linewidth': 2, 'marker': 'o', 'markersize': 3, 'label': 'YOLOv11'},
    'YOLOv12': {'color': '#F18F01', 'linestyle': '-.', 'linewidth': 2, 'marker': 's', 'markersize': 3, 'label': 'YOLOv12'},
    'YOLOv26': {'color': '#A23B72', 'linestyle': ':', 'linewidth': 2, 'marker': '^', 'markersize': 3, 'label': 'YOLOv26'},
}

# Эталонная линия (количество объектов в датасете)
GROUND_TRUTH_DETECTIONS = 324
GROUND_TRUTH_COLOR = 'black'
GROUND_TRUTH_LINESTYLE = '-'
GROUND_TRUTH_LINEWIDTH = 3


class YOLOPlotter:
    """Построение графиков сравнения YOLO моделей с эталоном"""
    
    def __init__(self):
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Загрузка данных из CSV"""
        if not RESULTS_FILE.exists():
            print(f"Файл {RESULTS_FILE} не найден")
            return
        
        df = pd.read_csv(RESULTS_FILE)
        self.models = {
            'YOLOv11': df[df['Model'] == 'YOLOv11'].copy(),
            'YOLOv12': df[df['Model'] == 'YOLOv12'].copy(),
            'YOLOv26': df[df['Model'] == 'YOLOv26'].copy()
        }
        
        for model_name, model_df in self.models.items():
            model_df = model_df.sort_values('Confidence_Threshold')
            self.models[model_name] = model_df
        
        print(f"Загружены данные: {len(self.models['YOLOv11'])} записей для YOLOv11")
        print(f"Загружены данные: {len(self.models['YOLOv12'])} записей для YOLOv12")
        print(f"Загружены данные: {len(self.models['YOLOv26'])} записей для YOLOv26")
        print(f"Эталонное количество объектов: {GROUND_TRUTH_DETECTIONS}")
    
    def smooth_curve(self, x, y, num_points=300):
        """Сглаживание кривой"""
        x_clean = x.values
        y_clean = y.values
        
        if len(x_clean) < 4:
            return x_clean, y_clean
        
        x_smooth = np.linspace(min(x_clean), max(x_clean), num_points)
        
        try:
            spline = UnivariateSpline(x_clean, y_clean, s=0.05 * len(x_clean))
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
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Эталонная линия
        ax.axhline(y=GROUND_TRUTH_DETECTIONS, 
                   color=GROUND_TRUTH_COLOR, 
                   linestyle=GROUND_TRUTH_LINESTYLE, 
                   linewidth=GROUND_TRUTH_LINEWIDTH,
                   label=f'Эталон ({GROUND_TRUTH_DETECTIONS} объектов)')
        
        # Список для хранения информации о пересечениях
        intersections = []
        
        # Данные моделей и точки пересечения
        for model_name, model_df in self.models.items():
            if len(model_df) == 0:
                continue
            
            x = model_df['Confidence_Threshold'].values
            y = model_df['Total_Detections'].values
            style = MODEL_STYLES[model_name]
            
            # Сглаженная кривая
            x_smooth, y_smooth = self.smooth_curve(model_df['Confidence_Threshold'], 
                                                   model_df['Total_Detections'])
            ax.plot(x_smooth, y_smooth, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=style['label'])
            
            # Нахождение точки пересечения с эталоном
            x_intersect = self.find_intersection_point(x_smooth, y_smooth, GROUND_TRUTH_DETECTIONS)
            
            if x_intersect is not None and 0 <= x_intersect <= 1:
                intersections.append((model_name, x_intersect, style['color']))
                # Точка пересечения
                ax.plot(x_intersect, GROUND_TRUTH_DETECTIONS, 
                        'o', color=style['color'], markersize=8, 
                        markeredgecolor='black', markeredgewidth=0.5)
                # Вертикальная линия
                ax.axvline(x=x_intersect, color=style['color'], linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Порог уверенности', fontsize=12)
        ax.set_ylabel('Количество обнаруженных объектов', fontsize=12)
        ax.set_title('Сравнение количества обнаруженных объектов с эталоном', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, GROUND_TRUTH_DETECTIONS + 50)
        
        # Добавление подписей порогов внизу графика (на оси X)
        # Создаём второй набор тиков
        current_xticks = list(ax.get_xticks())
        new_xticks = current_xticks.copy()
        new_xtick_labels = [f'{tick:.2f}' if tick not in [x for _, x, _ in intersections] else '' for tick in current_xticks]
        
        for model_name, x_int, color in intersections:
            if x_int not in new_xticks:
                new_xticks.append(x_int)
                new_xtick_labels.append(f'{x_int:.2f}')
            else:
                idx = new_xticks.index(x_int)
                new_xtick_labels[idx] = f'{x_int:.2f}'
        
        # Сортируем тики
        sorted_pairs = sorted(zip(new_xticks, new_xtick_labels))
        new_xticks = [p[0] for p in sorted_pairs]
        new_xtick_labels = [p[1] for p in sorted_pairs]
        
        ax.set_xticks(new_xticks)
        ax.set_xticklabels(new_xtick_labels, rotation=45, ha='right', fontsize=9)
        
        # Добавляем цветные метки для подписей
        for model_name, x_int, color in intersections:
            # Находим индекс в новых тиках
            if x_int in new_xticks:
                idx = new_xticks.index(x_int)
                # Изменяем цвет метки
                ax.get_xticklabels()[idx].set_color(color)
                ax.get_xticklabels()[idx].set_fontweight('bold')
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_detections_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_recall_comparison(self):
        """График: Recall@100 vs порог"""
        plt.figure(figsize=(10, 6))
        
        for model_name, model_df in self.models.items():
            if len(model_df) == 0:
                continue
            
            x = model_df['Confidence_Threshold']
            y = model_df['Recall@100']
            style = MODEL_STYLES[model_name]
            
            plt.plot(x, y, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    marker=style['marker'],
                    markersize=style['markersize'],
                    label=style['label'])
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('Recall@100 (%)', fontsize=12)
        plt.title('Сравнение полноты обнаружения', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_recall_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_map_comparison(self):
        """График: mAP@0.5 vs порог"""
        plt.figure(figsize=(10, 6))
        
        for model_name, model_df in self.models.items():
            if len(model_df) == 0:
                continue
            
            x = model_df['Confidence_Threshold']
            y = model_df['mAP@0.5']
            style = MODEL_STYLES[model_name]
            
            plt.plot(x, y, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    marker=style['marker'],
                    markersize=style['markersize'],
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
    
    def plot_map095_comparison(self):
        """График: mAP@0.5:0.95 vs порог"""
        plt.figure(figsize=(10, 6))
        
        for model_name, model_df in self.models.items():
            if len(model_df) == 0:
                continue
            
            x = model_df['Confidence_Threshold']
            y = model_df['mAP@0.5:0.95']
            style = MODEL_STYLES[model_name]
            
            plt.plot(x, y, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    marker=style['marker'],
                    markersize=style['markersize'],
                    label=style['label'])
        
        plt.xlabel('Порог уверенности', fontsize=12)
        plt.ylabel('mAP@0.5:0.95 (%)', fontsize=12)
        plt.title('Сравнение точности локализации', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        plt.tight_layout()
        output_path = RESULTS_FILE.parent / 'plot_map095_comparison.png'
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Сохранён: {output_path}")
    
    def plot_fps_comparison(self):
        """График: FPS vs порог"""
        plt.figure(figsize=(10, 6))
        
        for model_name, model_df in self.models.items():
            if len(model_df) == 0:
                continue
            
            x = model_df['Confidence_Threshold']
            y = model_df['FPS']
            style = MODEL_STYLES[model_name]
            
            plt.plot(x, y, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    marker=style['marker'],
                    markersize=style['markersize'],
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
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ YOLO МОДЕЛЕЙ".center(60))
        print("="*60 + "\n")
        
        self.plot_detections_comparison()
        self.plot_recall_comparison()
        self.plot_map_comparison()
        self.plot_map095_comparison()
        self.plot_fps_comparison()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


if __name__ == "__main__":
    plotter = YOLOPlotter()
    plotter.plot_all()