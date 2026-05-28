# detectors/yolo12_size_research_v2.py
"""Исследование влияния размера изображения на производительность YOLOv12 (изображения с шагом 10px)"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.paths import DETECTORS_DIR, REPO_ROOT, RUNS_DIR
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class YOLOv12SizeResearchV2:
    """Исследование влияния размера изображения на детекцию YOLOv12"""
    
    def __init__(self, model_path, images_dir, num_runs=10, warmup_runs=5):
        self.model = YOLO(model_path)
        self.images_dir = Path(images_dir)
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.threshold = 0.3
        
        self.results = []
    
    def get_image_files(self):
        """Получение всех изображений из папки и подпапок, сгруппированных по исходному изображению"""
        # Рекурсивно ищем все изображения во всех подпапках
        image_files = list(self.images_dir.rglob("*.jpg")) + \
                      list(self.images_dir.rglob("*.jpeg")) + \
                      list(self.images_dir.rglob("*.png"))
        
        # Группировка по исходному изображению (по префиксу до дефиса)
        grouped = {}
        for f in image_files:
            try:
                # Извлекаем префикс (номер изображения) из имени файла
                # Формат: "1-2048.jpg" или "1-8.jpg"
                filename = f.stem  # без расширения
                prefix = filename.split('-')[0]
                
                # Извлекаем высоту из имени файла
                height = int(filename.split('-')[1])
                
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append((f, height))
            except (IndexError, ValueError) as e:
                print(f"Предупреждение: файл {f.name} не соответствует формату, пропускаем ({e})")
                continue
        
        # Сортировка внутри каждой группы по высоте
        for prefix in grouped:
            grouped[prefix] = sorted(grouped[prefix], key=lambda x: x[1])  # сортируем по высоте
            # Оставляем только пути к файлам
            grouped[prefix] = [item[0] for item in grouped[prefix]]
        
        return grouped
    
    def get_image_size(self, img_path):
        """Получение размеров изображения"""
        with Image.open(img_path) as img:
            width, height = img.size
        return width, height
    
    def warmup_model(self, img_path):
        """Прогрев модели"""
        for _ in range(self.warmup_runs):
            _ = self.model(img_path, conf=self.threshold, verbose=False)
    
    def process_image(self, img_path):
        """Обработка одного изображения"""
        # Прогрев
        self.warmup_model(img_path)
        
        detections = []
        times = []
        confidences = []
        
        for _ in range(self.num_runs):
            start_time = time.perf_counter()
            results = self.model(img_path, conf=self.threshold, verbose=False)
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                num = len(result.boxes)
                avg_conf = result.boxes.conf.mean().item()
            else:
                num = 0
                avg_conf = 0.0
            
            detections.append(num)
            times.append(elapsed_time)
            confidences.append(avg_conf)
        
        return {
            'avg_detections': np.mean(detections),
            'std_detections': np.std(detections),
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences)
        }
    
    def run_research(self):
        """Запуск исследования"""
        print("="*70)
        print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ РАЗМЕРА ИЗОБРАЖЕНИЯ НА YOLOv12".center(70))
        print("="*70)
        print(f"\nМодель: YOLOv12")
        print(f"Порог уверенности: {self.threshold}")
        print(f"Прогрев: {self.warmup_runs} прогонов")
        print(f"Измерений: {self.num_runs} прогонов\n")
        
        grouped_images = self.get_image_files()
        
        print(f"Найдено исходных изображений: {len(grouped_images)}")
        print(f"Префиксы: {sorted(grouped_images.keys(), key=int)[:10]}...")
        
        # Общее количество файлов для обработки
        total_files = sum(len(images) for images in grouped_images.values())
        print(f"Всего файлов для обработки: {total_files}\n")
        
        processed = 0
        for prefix, images in sorted(grouped_images.items(), key=lambda x: int(x[0])):
            print(f"\n{'='*50}")
            print(f"Исходное изображение {prefix} (всего {len(images)} размеров)")
            print(f"{'='*50}")
            
            for img_path in tqdm(images, desc=f"  Обработка {prefix}"):
                width, height = self.get_image_size(img_path)
                size_mb = img_path.stat().st_size / (1024 * 1024)
                img_name = img_path.name
                
                result = self.process_image(img_path)
                
                self.results.append({
                    'image_prefix': prefix,
                    'image_name': img_name,
                    'width': width,
                    'height': height,
                    'pixels_mp': width * height / 1e6,
                    'file_size_mb': size_mb,
                    'avg_detections': result['avg_detections'],
                    'std_detections': result['std_detections'],
                    'avg_time_ms': result['avg_time_ms'],
                    'std_time_ms': result['std_time_ms'],
                    'avg_confidence': result['avg_confidence'],
                    'std_confidence': result['std_confidence']
                })
                
                processed += 1
                # Периодически выводим информацию (каждый 10-й файл)
                if processed % 10 == 0:
                    print(f"    {img_name}: {width}×{height} | "
                          f"Время: {result['avg_time_ms']:.1f}±{result['std_time_ms']:.1f} мс | "
                          f"Детекций: {result['avg_detections']:.0f} | "
                          f"Уверенность: {result['avg_confidence']:.3f}")
    
    def save_results_csv(self, output_path="yolo12_size_results_v2.csv"):
        df = pd.DataFrame(self.results)
        df = df.sort_values(['image_prefix', 'height'])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nРезультаты сохранены в {output_path}")
        print(f"Всего записей: {len(df)}")
        print(f"Количество уникальных изображений: {df['image_prefix'].nunique()}")
        print(f"Диапазон высот: {df['height'].min()} - {df['height'].max()}")
        return df


class ResultsPlotter:
    """Построение графиков по результатам"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # Определяем все уникальные префиксы
        self.all_prefixes = sorted(self.df['image_prefix'].unique(), key=lambda x: int(x))
        print(f"Найдено префиксов: {len(self.all_prefixes)}")
        print(f"Префиксы: {self.all_prefixes[:10]}..." if len(self.all_prefixes) > 10 else f"Префиксы: {self.all_prefixes}")
        
        # Усреднение по одинаковым высотам для каждого исходного изображения
        self.df_grouped = self.df.groupby(['image_prefix', 'height']).agg({
            'avg_time_ms': 'mean',
            'std_time_ms': 'mean',
            'avg_confidence': 'mean',
            'std_confidence': 'mean',
            'avg_detections': 'mean',
            'pixels_mp': 'mean'
        }).reset_index()
        
        self.df_grouped = self.df_grouped.sort_values('height')
        
        # Для усреднённого графика по всем изображениям
        self.df_overall = self.df.groupby('height').agg({
            'avg_time_ms': 'mean',
            'std_time_ms': 'mean',
            'avg_confidence': 'mean',
            'std_confidence': 'mean',
            'avg_detections': 'mean'
        }).reset_index().sort_values('height')
        
        print(f"Уникальных размеров по высоте: {len(self.df_overall)}")
        print(f"Диапазон высот: {self.df_overall['height'].min()} - {self.df_overall['height'].max()}")
    
    def get_colors_and_markers(self, num_prefixes):
        """Генерация цветов и маркеров для большого количества префиксов"""
        base_colors = [
            'blue', 'green', 'red', 'purple', 'orange', 
            'brown', 'pink', 'gray', 'olive', 'cyan',
            'navy', 'darkgreen', 'crimson', 'indigo', 'gold',
            'teal', 'salmon', 'royalblue', 'orchid', 'slategray'
        ]
        base_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 
                        'H', '+', 'x', 'd', '|', '_', 'P', 'X', '8', '1']
        
        colors = {}
        markers = {}
        for i, prefix in enumerate(self.all_prefixes):
            colors[prefix] = base_colors[i % len(base_colors)]
            markers[prefix] = base_markers[i % len(base_markers)]
        
        return colors, markers
    
    def plot_all_summary(self):
        """Сводный график усреднённых данных"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = self.df_overall['height'].values
        
        # 1. Время обработки
        ax1 = axes[0, 0]
        ax1.errorbar(x, self.df_overall['avg_time_ms'], 
                    yerr=self.df_overall['std_time_ms'],
                    fmt='o-', color='blue', capsize=3, markersize=4)
        ax1.set_xlabel('Высота изображения (пиксели)')
        ax1.set_ylabel('Время (мс)')
        ax1.set_title('Время обработки')
        ax1.grid(True, alpha=0.3)
        
        # 2. Уверенность
        ax2 = axes[0, 1]
        ax2.errorbar(x, self.df_overall['avg_confidence'], 
                    yerr=self.df_overall['std_confidence'],
                    fmt='o-', color='orange', capsize=3, markersize=4)
        ax2.set_xlabel('Высота изображения (пиксели)')
        ax2.set_ylabel('Уверенность')
        ax2.set_title('Уверенность детекции')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. Количество детекций
        ax3 = axes[1, 0]
        ax3.errorbar(x, self.df_overall['avg_detections'], 
                    yerr=self.df_overall['std_detections'],
                    fmt='o-', color='green', capsize=3, markersize=4)
        ax3.set_xlabel('Высота изображения (пиксели)')
        ax3.set_ylabel('Количество')
        ax3.set_title('Количество детекций')
        ax3.grid(True, alpha=0.3)
        
        # 4. Соотношение время/уверенность
        ax4 = axes[1, 1]
        scatter = ax4.scatter(self.df_overall['avg_time_ms'], 
                             self.df_overall['avg_confidence'],
                             c=x, cmap='viridis', s=50, alpha=0.7)
        ax4.set_xlabel('Время обработки (мс)')
        ax4.set_ylabel('Уверенность')
        ax4.set_title('Время vs Уверенность (цвет = высота)')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Высота (пиксели)')
        
        plt.suptitle(f'Сводная статистика (усреднённо по {len(self.all_prefixes)} изображениям)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'plot_all_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_all_summary.png")
    
    def generate_statistics(self):
        """Генерация статистики по данным"""
        print("\n" + "="*60)
        print("СТАТИСТИКА ПО ДАННЫМ".center(60))
        print("="*60)
        
        df = self.df_overall
        
        # Где максимальная уверенность
        max_conf_idx = df['avg_confidence'].idxmax()
        max_conf_height = df.loc[max_conf_idx, 'height']
        max_conf_value = df.loc[max_conf_idx, 'avg_confidence']
        
        # Где минимальное время
        min_time_idx = df['avg_time_ms'].idxmin()
        min_time_height = df.loc[min_time_idx, 'height']
        min_time_value = df.loc[min_time_idx, 'avg_time_ms']
        
        # Где максимальное количество детекций
        max_det_idx = df['avg_detections'].idxmax()
        max_det_height = df.loc[max_det_idx, 'height']
        max_det_value = df.loc[max_det_idx, 'avg_detections']
        
        print(f"\n📊 Оптимальные значения (усреднённые по {len(self.all_prefixes)} изображениям):")
        print(f"  • Максимальная уверенность: {max_conf_value:.3f} при высоте {max_conf_height:.0f}px")
        print(f"  • Минимальное время обработки: {min_time_value:.1f} мс при высоте {min_time_height:.0f}px")
        print(f"  • Максимальное количество детекций: {max_det_value:.0f} при высоте {max_det_height:.0f}px")
        
        # Статистика по уверенности
        print(f"\n📊 Дополнительная статистика:")
        print(f"  • Средняя уверенность по всем размерам: {df['avg_confidence'].mean():.3f} ± {df['avg_confidence'].std():.3f}")
        print(f"  • Среднее время обработки: {df['avg_time_ms'].mean():.1f} ± {df['avg_time_ms'].std():.1f} мс")
        print(f"  • Среднее количество детекций: {df['avg_detections'].mean():.1f} ± {df['avg_detections'].std():.1f}")
        
        return {
            'max_confidence': (max_conf_height, max_conf_value),
            'min_time': (min_time_height, min_time_value),
            'max_detections': (max_det_height, max_det_value)
        }
    
    def plot_all(self):
        """Построение всех графиков"""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_all_summary()
        self.generate_statistics()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


def main():
    # Поиск модели
    MODEL_PATH = DETECTORS_DIR / "yolo12" / "runs" / "train" / "weights" / "best.pt"
    
    if not MODEL_PATH.exists():
        alt_paths = [
            RUNS_DIR / "detect" / "train" / "weights" / "best.pt",
            REPO_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
            Path(__file__).parent.parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
        for p in alt_paths:
            if p.exists():
                MODEL_PATH = p
                break
    
    if not MODEL_PATH.exists():
        print(f"Ошибка: модель не найдена")
        return
    
    print(f"Модель загружена: {MODEL_PATH}")
    
    # Папка с изображениями (где лежат подпапки height_*)
    # Путь к корневой папке с изображениями
    IMAGES_DIR = Path(__file__).parent / "test_diff_sizes_imgs_resized"
    
    if not IMAGES_DIR.exists():
        # Пробуем альтернативные пути
        alt_images = [
            Path(__file__).parent / "test_diff_sizes_imgs",
            Path(__file__).parent.parent / "test_diff_sizes_imgs_resized",
        ]
        for p in alt_images:
            if p.exists():
                IMAGES_DIR = p
                break
    
    if not IMAGES_DIR.exists():
        print(f"Ошибка: папка с изображениями не найдена")
        print(f"Искали в: {Path(__file__).parent / 'test_diff_sizes_imgs_resized'}")
        return
    
    print(f"Папка с изображениями: {IMAGES_DIR}")
    
    # Подсчёт количества изображений во всех подпапках
    image_count = len(list(IMAGES_DIR.rglob("*.jpg"))) + len(list(IMAGES_DIR.rglob("*.png")))
    print(f"Найдено изображений: {image_count}")
    
    # Запуск исследования
    researcher = YOLOv12SizeResearchV2(
        model_path=str(MODEL_PATH),
        images_dir=str(IMAGES_DIR),
        num_runs=3,      # 3 прогона для ускорения
        warmup_runs=2    # 2 прогрева
    )
    
    researcher.run_research()
    df = researcher.save_results_csv("yolo12_size_results_v2.csv")
    
    # Построение графиков
    plotter = ResultsPlotter("yolo12_size_results_v2.csv")
    plotter.plot_all()


if __name__ == "__main__":
    main()