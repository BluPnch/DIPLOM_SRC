# detectors/plot_size_research_fixed.py
"""Построение графиков зависимости производительности от размера изображения с аппроксимацией"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from pathlib import Path

# Путь к файлу с результатами
RESULTS_FILE = Path(__file__).parent / "yolo12_size_results_v2.csv"
OUTPUT_DIR = Path(__file__).parent


class SizeResearchPlotter:
    """Построение графиков исследования влияния размера изображения"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Удаляем дубликаты, оставляя среднее по одинаковым высотам
        self.df = self.df.groupby('height').agg({
            'avg_time_ms': 'mean',
            'std_time_ms': 'mean',
            'avg_detections': 'mean',
            'std_detections': 'mean',
            'avg_confidence': 'mean',
            'std_confidence': 'mean'
        }).reset_index()
        
        self.df = self.df.sort_values('height')
        
        print(f"Уникальных размеров по высоте: {len(self.df)}")
        print(f"Высоты: {list(self.df['height'].astype(int))}")
    
    def interpolate_curve(self, x, y, num_points=200, smoothing=0.1):
        """Создание сглаженной кривой"""
        mask = ~np.isnan(y)
        x_clean = np.array(x)[mask]
        y_clean = np.array(y)[mask]

        if len(x_clean) < 2:
            return x_clean, y_clean

        x_smooth = np.linspace(min(x_clean), max(x_clean), num_points)

        try:
            interpolator = PchipInterpolator(x_clean, y_clean)
            y_smooth = interpolator(x_smooth)
        except:
            y_smooth = np.interp(x_smooth, x_clean, y_clean)

        return x_smooth, y_smooth
    
    def plot_time_vs_height(self):
        """График: Время обработки от высоты изображения"""
        plt.figure(figsize=(12, 7))
        
        x = self.df['height'].values
        y = self.df['avg_time_ms'].values
        y_err = self.df['std_time_ms'].values
        
        plt.errorbar(x, y, yerr=y_err, fmt='o', color='blue', 
                    capsize=5, capthick=1, markersize=8, 
                    label='Экспериментальные данные', alpha=0.8)
        
        x_smooth, y_smooth = self.interpolate_curve(x, y, smoothing=0.1)
        plt.plot(x_smooth, y_smooth, '-', color='red', linewidth=2.5, 
                label='Аппроксимация (сплайн)')
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Время обработки (мс)', fontsize=12)
        plt.title('Зависимость времени обработки от высоты изображения', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Только положительные значения
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_time_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_time_vs_height.png")
    
    def plot_detections_vs_height(self):
        """График: Количество детекций от высоты"""
        plt.figure(figsize=(12, 7))
        
        x = self.df['height'].values
        y = self.df['avg_detections'].values
        y_err = self.df['std_detections'].values
        
        plt.errorbar(x, y, yerr=y_err, fmt='s', color='green', 
                    capsize=5, capthick=1, markersize=8, 
                    label='Экспериментальные данные', alpha=0.8)
        
        x_smooth, y_smooth = self.interpolate_curve(x, y, smoothing=0.1)
        plt.plot(x_smooth, y_smooth, '-', color='darkgreen', linewidth=2.5, 
                label='Аппроксимация (сплайн)')
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Количество детекций', fontsize=12)
        plt.title('Зависимость количества обнаруженных объектов от высоты', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_detections_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_detections_vs_height.png")
    
    def plot_confidence_vs_height(self):
        """График: Уверенность от высоты"""
        plt.figure(figsize=(12, 7))
        
        x = self.df['height'].values
        y = self.df['avg_confidence'].values
        y_err = self.df['std_confidence'].values
        
        plt.errorbar(x, y, yerr=y_err, fmt='^', color='orange', 
                    capsize=5, capthick=1, markersize=8, 
                    label='Экспериментальные данные', alpha=0.8)
        
        x_smooth, y_smooth = self.interpolate_curve(x, y, smoothing=0.1)
        plt.plot(x_smooth, y_smooth, '-', color='darkorange', linewidth=2.5, 
                label='Аппроксимация (сплайн)')
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Средняя уверенность', fontsize=12)
        plt.title('Зависимость средней уверенности от высоты изображения', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_confidence_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_confidence_vs_height.png")
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_time_vs_height()
        self.plot_detections_vs_height()
        self.plot_confidence_vs_height()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


def main():
    if not RESULTS_FILE.exists():
        print(f"Ошибка: файл {RESULTS_FILE} не найден")
        print("Сначала запустите yolo12_size_research_fixed.py")
        return
    
    plotter = SizeResearchPlotter(RESULTS_FILE)
    plotter.plot_all()


if __name__ == "__main__":
    main()