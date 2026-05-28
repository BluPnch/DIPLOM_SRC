# detectors/plot_size_research_fixed.py
"""Построение графиков зависимости производительности от размера изображения (БЕЗ аппроксимации)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg') 

# Путь к файлу с результатами
RESULTS_FILE = Path(__file__).parent / "yolo12_size_results_v2.csv"
OUTPUT_DIR = Path(__file__).parent


class SizeResearchPlotter:
    """Построение графиков исследования влияния размера изображения (только экспериментальные данные)"""
    
    def __init__(self, csv_path, show_individual=False, max_individual_lines=10):
        """
        Args:
            csv_path: путь к CSV файлу с результатами
            show_individual: показывать ли графики отдельных изображений
            max_individual_lines: максимальное количество линий на графике отдельных изображений
        """
        self.show_individual = show_individual
        self.max_individual_lines = max_individual_lines
        
        self.df = pd.read_csv(csv_path)
        
        # Сохраняем информацию о разных изображениях
        self.df_raw = self.df.copy()
        
        # Для общего графика усредняем по высоте (но сохраняем std)
        self.df_grouped = self.df.groupby('height').agg({
            'avg_time_ms': 'mean',
            'std_time_ms': 'mean',
            'avg_detections': 'mean',
            'std_detections': 'mean',
            'avg_confidence': 'mean',
            'std_confidence': 'mean'
        }).reset_index()
        
        self.df_grouped = self.df_grouped.sort_values('height')
        
        # Удаляем выбросы на графике скорости
        self.df_grouped = self._remove_outliers_time(self.df_grouped)
        
        # Группируем по префиксам для отдельных графиков
        self.df_by_prefix = {}
        self.all_prefixes = []
        for prefix in self.df['image_prefix'].unique():
            prefix_str = str(prefix)
            self.all_prefixes.append(prefix_str)
            self.df_by_prefix[prefix_str] = self.df[self.df['image_prefix'] == prefix].sort_values('height')
        
        # Сортируем префиксы по числовому значению
        self.all_prefixes = sorted(self.all_prefixes, key=lambda x: int(x))
        
        # Генерируем цвета и маркеры
        self.colors, self.markers = self._generate_colors_and_markers(len(self.all_prefixes))
        
        print(f"Уникальных размеров по высоте: {len(self.df_grouped)}")
        print(f"Диапазон высот: {self.df_grouped['height'].min():.0f} - {self.df_grouped['height'].max():.0f}")
        print(f"Количество исходных изображений: {len(self.all_prefixes)}")
        print(f"Всего измерений: {len(self.df)}")
        print(f"Префиксы изображений (первые 10): {self.all_prefixes[:10]}...")
        
        # Находим порог 80% уверенности
        self._find_80_percent_threshold()
        
        # Статистика по данным
        self._print_data_stats()
    
    def _remove_outliers_time(self, df):
        """Удаление выбросов на графике времени обработки"""
        Q1 = df['avg_time_ms'].quantile(0.25)
        Q3 = df['avg_time_ms'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['avg_time_ms'] < lower_bound) | (df['avg_time_ms'] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"\n⚠️ Обнаружены выбросы на графике времени ({len(outliers)} точек):")
            for idx, row in outliers.iterrows():
                print(f"   • Высота {row['height']:.0f}px: время = {row['avg_time_ms']:.1f} мс")
            
            for idx in outliers.index:
                height_val = df.loc[idx, 'height']
                nearby = df[(df['height'] > height_val - 100) & (df['height'] < height_val + 100) & (df.index != idx)]
                if len(nearby) > 0:
                    new_value = nearby['avg_time_ms'].mean()
                    print(f"   → Заменяем на {new_value:.1f} мс (среднее соседних)")
                    df.loc[idx, 'avg_time_ms'] = new_value
        
        return df
    
    def _find_80_percent_threshold(self):
        """Находит первую высоту, где уверенность превышает 80%"""
        df_sorted = self.df_grouped.sort_values('height')
        
        mask = df_sorted['avg_confidence'] > 0.8
        if mask.any():
            first_idx = mask.idxmax()
            self.threshold_height = df_sorted.loc[first_idx, 'height']
            self.threshold_confidence = df_sorted.loc[first_idx, 'avg_confidence']
            self.threshold_time = df_sorted.loc[first_idx, 'avg_time_ms']
            
            print(f"\n🎯 80% порог уверенности достигнут при высоте {self.threshold_height:.0f}px")
            print(f"   Уверенность: {self.threshold_confidence:.3f}")
            print(f"   Время обработки: {self.threshold_time:.1f} мс")
        else:
            self.threshold_height = None
            self.threshold_confidence = None
            self.threshold_time = None
            print("\n⚠️ Порог 80% уверенности не достигнут ни на одной высоте")
    
    def _print_data_stats(self):
        """Вывод базовой статистики по данным"""
        print("\n" + "="*60)
        print("БАЗОВАЯ СТАТИСТИКА ДАННЫХ".center(60))
        print("="*60)
        print(f"  • Количество уникальных высот: {self.df_grouped['height'].nunique()}")
        print(f"  • Минимальная высота: {self.df_grouped['height'].min():.0f} px")
        print(f"  • Максимальная высота: {self.df_grouped['height'].max():.0f} px")
        print(f"  • Среднее время обработки: {self.df_grouped['avg_time_ms'].mean():.1f} ± {self.df_grouped['avg_time_ms'].std():.1f} мс")
        print(f"  • Средняя уверенность: {self.df_grouped['avg_confidence'].mean():.3f} ± {self.df_grouped['avg_confidence'].std():.3f}")
        print(f"  • Среднее количество детекций: {self.df_grouped['avg_detections'].mean():.1f} ± {self.df_grouped['avg_detections'].std():.1f}")
    
    def _generate_colors_and_markers(self, num_prefixes):
        """Генерация цветов и маркеров для большого количества префиксов"""
        base_colors = [
            'blue', 'green', 'red', 'purple', 'orange', 
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'navy', 'darkgreen', 'crimson', 'indigo', 'gold',
            'teal', 'salmon', 'royalblue', 'orchid', 'slategray'
        ]
        
        base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        colors = {}
        markers = {}
        for i, prefix in enumerate(self.all_prefixes):
            colors[prefix] = base_colors[i % len(base_colors)]
            markers[prefix] = base_markers[i % len(base_markers)]
        
        return colors, markers
    
    def plot_time_vs_height(self):
        """График: Время обработки от высоты изображения (без усиков)"""
        if self.show_individual:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax_left = axes[0]
            ax_right = axes[1]
        else:
            fig, ax_left = plt.subplots(1, 1, figsize=(12, 8))
        
        # === Левый график: общий, только средние значения ===
        x = self.df_grouped['height'].values
        y = self.df_grouped['avg_time_ms'].values
        
        # Только точки, БЕЗ error bars
        ax_left.plot(x, y, 'o', color='blue', markersize=6, alpha=0.8, zorder=1)
        
        ax_left.set_xlabel('Высота изображения (пиксели)', fontsize=16)
        ax_left.set_ylabel('Время обработки (мс)', fontsize=16)
        ax_left.set_title('Время обработки (усреднённое по всем изображениям)', fontsize=16, fontweight='bold')
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_ylim(bottom=0)
        
        # === Линии к осям для порога 80% ===
        if self.threshold_height is not None:
            # Вертикальная линия от точки к оси X
            ax_left.axvline(x=self.threshold_height, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5)
            # Горизонтальная линия от точки к оси Y
            ax_left.axhline(y=self.threshold_time, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5)
            # Точка пересечения
            ax_left.plot(self.threshold_height, self.threshold_time, 'ro', 
                        markersize=8, zorder=5)
            ax_left.text(self.threshold_height - 50, -0.08 * max(y), f'{self.threshold_height:.0f}px', 
                        color='red', fontsize=12, ha='center', va='top')
            ax_left.text(-0.08 * max(x), self.threshold_time + 5, f'{self.threshold_time:.1f}мс', 
                        color='red', fontsize=12, ha='right', va='center')
        
        # === Правый график (опционально): по отдельным изображениям ===
        if self.show_individual:
            prefixes_to_show = self.all_prefixes[:self.max_individual_lines]
            
            for prefix_str in prefixes_to_show:
                data = self.df_by_prefix.get(prefix_str)
                if data is not None and len(data) > 0:
                    x = data['height'].values
                    y = data['avg_time_ms'].values
                    
                    marker_style = self.markers.get(prefix_str, 'o')
                    color_style = self.colors.get(prefix_str, 'gray')
                    
                    ax_right.plot(x, y, marker_style, color=color_style, 
                                markersize=4, linestyle='none',
                                label=f'{prefix_str}', alpha=0.7)
            
            ax_right.set_xlabel('Высота изображения (пиксели)', fontsize=14)
            ax_right.set_ylabel('Время обработки (мс)', fontsize=14)
            ax_right.set_title(f'Время обработки (первые {len(prefixes_to_show)} изображений)', fontsize=14, fontweight='bold')
            ax_right.legend(loc='upper left', fontsize=8, ncol=2)
            ax_right.grid(True, alpha=0.3, linestyle='--')
            ax_right.set_ylim(bottom=0)
            
            if self.threshold_height is not None:
                ax_right.axvline(x=self.threshold_height, color='red', linestyle='--', 
                               alpha=0.7, linewidth=1.5)
        
        plt.suptitle('Зависимость времени обработки от высоты изображения', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_time_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_time_vs_height.png")
    
    def plot_detections_vs_height(self):
        """График: Количество детекций от высоты (БЕЗ усиков)"""
        if self.show_individual:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax_left = axes[0]
            ax_right = axes[1]
        else:
            fig, ax_left = plt.subplots(1, 1, figsize=(12, 8))
        
        # === Левый график: усреднённый, только точки ===
        x = self.df_grouped['height'].values
        y = self.df_grouped['avg_detections'].values
        
        # Только точки, БЕЗ error bars
        ax_left.plot(x, y, 's', color='green', markersize=5, alpha=0.8)
        
        ax_left.set_xlabel('Высота изображения (пиксели)', fontsize=16)
        ax_left.set_ylabel('Количество детекций', fontsize=16)
        ax_left.set_title('Количество детекций (усреднённое)', fontsize=16, fontweight='bold')
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_ylim(bottom=0)
        
        # === Вертикальная линия для порога 80% ===
        if self.threshold_height is not None:
            ax_left.axvline(x=self.threshold_height, color='red', linestyle='--', 
                           alpha=0.7, linewidth=1.5)
        
        # === Правый график: отдельные изображения ===
        if self.show_individual:
            prefixes_to_show = self.all_prefixes[:self.max_individual_lines]
            
            for prefix_str in prefixes_to_show:
                data = self.df_by_prefix.get(prefix_str)
                if data is not None and len(data) > 0:
                    x = data['height'].values
                    y = data['avg_detections'].values
                    
                    marker_style = self.markers.get(prefix_str, 'o')
                    color_style = self.colors.get(prefix_str, 'gray')
                    
                    ax_right.plot(x, y, marker_style, color=color_style, 
                                markersize=4, linestyle='none',
                                label=f'{prefix_str}', alpha=0.7)
            
            ax_right.set_xlabel('Высота изображения (пиксели)', fontsize=14)
            ax_right.set_ylabel('Количество детекций', fontsize=14)
            ax_right.set_title(f'Количество детекций (первые {len(prefixes_to_show)} изображений)', fontsize=14, fontweight='bold')
            ax_right.legend(loc='best', fontsize=8, ncol=2)
            ax_right.grid(True, alpha=0.3, linestyle='--')
            ax_right.set_ylim(bottom=0)
            
            if self.threshold_height is not None:
                ax_right.axvline(x=self.threshold_height, color='red', linestyle='--', 
                               alpha=0.7, linewidth=1.5)
        
        plt.suptitle('Зависимость количества обнаруженных объектов от высоты', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_detections_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_detections_vs_height.png")
    
    def plot_confidence_vs_height(self):
        """График: Уверенность от высоты с линиями к осям (БЕЗ усиков и плашек внутри)"""
        if self.show_individual:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            ax_left = axes[0]
            ax_right = axes[1]
        else:
            fig, ax_left = plt.subplots(1, 1, figsize=(12, 8))
        
        # === Левый график: усреднённый, только точки ===
        x = self.df_grouped['height'].values
        y = self.df_grouped['avg_confidence'].values
        
        # Только точки, БЕЗ error bars
        ax_left.plot(x, y, '^', color='orange', markersize=5, alpha=0.8)
        
        ax_left.set_xlabel('Высота изображения (пиксели)', fontsize=16)
        ax_left.set_ylabel('Средняя уверенность', fontsize=16)
        ax_left.set_title('Уверенность детекции (усреднённая)', fontsize=16, fontweight='bold')
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_ylim(0, 1)
        
        # === Линии к осям при достижении 80% (без подписей-плашек) ===
        if self.threshold_height is not None:
            # Горизонтальная линия на уровне 0.8
            ax_left.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Вертикальная линия на высоте порога
            ax_left.axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Линии от точки к осям (пунктирные)
            ax_left.plot([self.threshold_height, self.threshold_height], [0, self.threshold_confidence], 
                        'r--', alpha=0.4, linewidth=1)
            ax_left.plot([0, self.threshold_height], [self.threshold_confidence, self.threshold_confidence], 
                        'r--', alpha=0.4, linewidth=1)
            
            # Точка пересечения (без аннотации)
            ax_left.plot(self.threshold_height, self.threshold_confidence, 'ro', markersize=8, zorder=5)
            
            # Подписи на осях (значения)
            ax_left.text(self.threshold_height - 50, -0.03, f'{self.threshold_height:.0f}px', 
                        color='red', fontsize=12, ha='center', va='top')
            ax_left.text(-0.08 * max(x), self.threshold_confidence - 0.02, f'{self.threshold_confidence:.3f}', 
                        color='red', fontsize=12, ha='right', va='center')
        
        # === Правый график: отдельные изображения ===
        if self.show_individual:
            prefixes_to_show = self.all_prefixes[:self.max_individual_lines]
            
            for prefix_str in prefixes_to_show:
                data = self.df_by_prefix.get(prefix_str)
                if data is not None and len(data) > 0:
                    x = data['height'].values
                    y = data['avg_confidence'].values
                    
                    marker_style = self.markers.get(prefix_str, 'o')
                    color_style = self.colors.get(prefix_str, 'gray')
                    
                    ax_right.plot(x, y, marker_style, color=color_style, 
                                markersize=4, linestyle='none',
                                label=f'{prefix_str}', alpha=0.7)
            
            ax_right.set_xlabel('Высота изображения (пиксели)', fontsize=14)
            ax_right.set_ylabel('Средняя уверенность', fontsize=14)
            ax_right.set_title(f'Уверенность детекции (первые {len(prefixes_to_show)} изображений)', fontsize=14, fontweight='bold')
            ax_right.legend(loc='best', fontsize=8, ncol=2)
            ax_right.grid(True, alpha=0.3, linestyle='--')
            ax_right.set_ylim(0, 1)
            
            if self.threshold_height is not None:
                ax_right.axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax_right.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.suptitle('Зависимость средней уверенности от высоты изображения', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_confidence_vs_height.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_confidence_vs_height.png")
    
    def plot_combined_metrics(self):
        """Сводный график всех метрик (усреднённых) - БЕЗ усиков"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = self.df_grouped['height'].values
        
        # Время - только точки
        axes[0].plot(x, self.df_grouped['avg_time_ms'], 'o', color='blue', markersize=5)
        axes[0].set_xlabel('Высота (пиксели)', fontsize=14)
        axes[0].set_ylabel('Время (мс)', fontsize=14)
        axes[0].set_title('Время обработки', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        if self.threshold_height is not None:
            axes[0].axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Уверенность - только точки
        axes[1].plot(x, self.df_grouped['avg_confidence'], 'o', color='orange', markersize=5)
        axes[1].set_xlabel('Высота (пиксели)', fontsize=14)
        axes[1].set_ylabel('Уверенность', fontsize=14)
        axes[1].set_title('Уверенность детекции', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        if self.threshold_height is not None:
            axes[1].axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Детекции - только точки
        axes[2].plot(x, self.df_grouped['avg_detections'], 'o', color='green', markersize=5)
        axes[2].set_xlabel('Высота (пиксели)', fontsize=14)
        axes[2].set_ylabel('Количество', fontsize=14)
        axes[2].set_title('Количество детекций', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        if self.threshold_height is not None:
            axes[2].axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.suptitle(f'Сравнение метрик (усреднённо по {len(self.all_prefixes)} изображениям)', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_combined_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_combined_metrics.png")
    
    def plot_relative_performance(self):
        """График относительной производительности - только точки"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = self.df_grouped['height'].values
        time_norm = self.df_grouped['avg_time_ms'].values / self.df_grouped['avg_time_ms'].max()
        conf_norm = self.df_grouped['avg_confidence'].values / self.df_grouped['avg_confidence'].max()
        det_norm = self.df_grouped['avg_detections'].values / self.df_grouped['avg_detections'].max()
        
        ax.plot(x, time_norm, 'o', color='red', markersize=5, label='Время обработки (норм.)', alpha=0.7)
        ax.plot(x, conf_norm, 's', color='blue', markersize=5, label='Уверенность (норм.)', alpha=0.7)
        ax.plot(x, det_norm, '^', color='green', markersize=5, label='Количество детекций (норм.)', alpha=0.7)
        
        ax.set_xlabel('Высота изображения (пиксели)', fontsize=16)
        ax.set_ylabel('Нормированное значение (0-1)', fontsize=16)
        ax.set_title(f'Относительная производительность (n={len(self.all_prefixes)} изображений)', 
                    fontsize=18, fontweight='bold')
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        if self.threshold_height is not None:
            ax.axvline(x=self.threshold_height, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_relative_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_relative_performance.png")
    
    def plot_heatmap(self, max_images=50):
        """Тепловая карта уверенности для всех изображений"""
        pivot_data = self.df.pivot_table(
            index='image_prefix', 
            columns='height', 
            values='avg_confidence',
            aggfunc='mean'
        )
        
        pivot_data = pivot_data.reindex(sorted(pivot_data.index, key=lambda x: int(x)))
        pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        
        if len(pivot_data) > max_images:
            step = len(pivot_data) // max_images
            pivot_data = pivot_data.iloc[::step]
            print(f"  Отображается {len(pivot_data)} из {len(self.all_prefixes)} изображений (шаг {step})")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis', 
                      interpolation='nearest', vmin=0, vmax=1)
        
        heights = pivot_data.columns
        step_x = max(1, len(heights) // 15)
        ax.set_xticks(range(0, len(heights), step_x))
        ax.set_xticklabels([f'{int(h)}' for h in heights[::step_x]], rotation=45)
        
        images = pivot_data.index
        step_y = max(1, len(images) // 20)
        ax.set_yticks(range(0, len(images), step_y))
        ax.set_yticklabels(images[::step_y])
        
        ax.set_xlabel('Высота изображения (пиксели)', fontsize=16)
        ax.set_ylabel('Номер изображения', fontsize=16)
        ax.set_title(f'Тепловая карта уверенности детекции (n={len(self.all_prefixes)} изображений)', 
                    fontsize=18, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Уверенность', fontsize=14)
        
        if self.threshold_height is not None:
            heights_list = list(heights)
            if self.threshold_height in heights_list:
                idx = heights_list.index(self.threshold_height)
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_heatmap.png")
    
    def plot_statistics_boxplot(self):
        """Boxplot распределения метрик по диапазонам высот"""
        self.df['height_group'] = pd.cut(self.df['height'], 
                                         bins=10, 
                                         labels=['≤200', '200-400', '400-600', '600-800', '800-1000',
                                                '1000-1200', '1200-1400', '1400-1600', '1600-1800', '1800-2048'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        self.df.boxplot(column='avg_time_ms', by='height_group', ax=axes[0])
        axes[0].set_title('Время обработки по группам высот', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Группа высот (пиксели)', fontsize=11)
        axes[0].set_ylabel('Время (мс)', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        
        self.df.boxplot(column='avg_confidence', by='height_group', ax=axes[1])
        axes[1].set_title('Уверенность по группам высот', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Группа высот (пиксели)', fontsize=11)
        axes[1].set_ylabel('Уверенность', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)
        
        self.df.boxplot(column='avg_detections', by='height_group', ax=axes[2])
        axes[2].set_title('Количество детекций по группам высот', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Группа высот (пиксели)', fontsize=11)
        axes[2].set_ylabel('Количество', fontsize=11)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Распределение метрик по диапазонам высот', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'plot_statistics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_statistics_boxplot.png")
    
    def generate_statistics(self):
        """Генерация полной статистики по данным"""
        print("\n" + "="*60)
        print("ПОДРОБНАЯ СТАТИСТИКА".center(60))
        print("="*60)
        
        df = self.df_grouped
        
        max_conf_idx = df['avg_confidence'].idxmax()
        max_conf_height = df.loc[max_conf_idx, 'height']
        max_conf_value = df.loc[max_conf_idx, 'avg_confidence']
        
        min_time_idx = df['avg_time_ms'].idxmin()
        min_time_height = df.loc[min_time_idx, 'height']
        min_time_value = df.loc[min_time_idx, 'avg_time_ms']
        
        max_det_idx = df['avg_detections'].idxmax()
        max_det_height = df.loc[max_det_idx, 'height']
        max_det_value = df.loc[max_det_idx, 'avg_detections']
        
        print(f"\n📊 Оптимальные значения (усреднённые по {len(self.all_prefixes)} изображениям):")
        print(f"  • Максимальная уверенность: {max_conf_value:.3f} при высоте {max_conf_height:.0f}px")
        print(f"  • Минимальное время обработки: {min_time_value:.1f} мс при высоте {min_time_height:.0f}px")
        print(f"  • Максимальное количество детекций: {max_det_value:.0f} при высоте {max_det_height:.0f}px")
        
        if self.threshold_height is not None:
            print(f"\n🎯 Порог 80% уверенности:")
            print(f"  • Достигнут при высоте: {self.threshold_height:.0f}px")
            print(f"  • Уверенность: {self.threshold_confidence:.3f}")
            print(f"  • Время обработки: {self.threshold_time:.1f} мс")
        
        try:
            from scipy.stats import pearsonr
            corr_time, p_time = pearsonr(df['height'], df['avg_time_ms'])
            corr_conf, p_conf = pearsonr(df['height'], df['avg_confidence'])
            corr_det, p_det = pearsonr(df['height'], df['avg_detections'])
            
            print(f"\n📈 Корреляции с высотой:")
            print(f"  • Время: r = {corr_time:.3f} (p={p_time:.2e})")
            print(f"  • Уверенность: r = {corr_conf:.3f} (p={p_conf:.2e})")
            print(f"  • Детекции: r = {corr_det:.3f} (p={p_det:.2e})")
        except:
            pass
        
        print(f"\n📊 Общая статистика:")
        print(f"  • Средняя уверенность: {df['avg_confidence'].mean():.3f} ± {df['avg_confidence'].std():.3f}")
        print(f"  • Среднее время: {df['avg_time_ms'].mean():.1f} ± {df['avg_time_ms'].std():.1f} мс")
        print(f"  • Среднее кол-во детекций: {df['avg_detections'].mean():.1f} ± {df['avg_detections'].std():.1f}")
        
        print(f"\n📊 95-й перцентили:")
        print(f"  • Время: {df['avg_time_ms'].quantile(0.95):.1f} мс")
        print(f"  • Уверенность: {df['avg_confidence'].quantile(0.95):.3f}")
        print(f"  • Детекции: {df['avg_detections'].quantile(0.95):.1f}")
        
        return {
            'max_confidence': (max_conf_height, max_conf_value),
            'min_time': (min_time_height, min_time_value),
            'max_detections': (max_det_height, max_det_value),
            'threshold_80': (self.threshold_height, self.threshold_confidence, self.threshold_time)
        }
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_time_vs_height()
        self.plot_detections_vs_height()
        self.plot_confidence_vs_height()
        self.plot_combined_metrics()
        self.plot_relative_performance()
        self.plot_heatmap(max_images=40)
        self.plot_statistics_boxplot()
        self.generate_statistics()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


def main():
    if not RESULTS_FILE.exists():
        print(f"Ошибка: файл {RESULTS_FILE} не найден")
        print("Сначала запустите yolo12_size_research_v2.py для получения результатов")
        return
    
    plotter = SizeResearchPlotter(
        RESULTS_FILE, 
        show_individual=False,
        max_individual_lines=10
    )
    plotter.plot_all()


if __name__ == "__main__":
    main()