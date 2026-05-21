# detectors/yolo12_size_research_v2.py
"""Исследование влияния размера изображения на производительность YOLOv12 (3 исходных изображения в разных разрешениях)"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
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
        """Получение всех изображений из папки, сгруппированных по исходному изображению"""
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.jpeg")) + list(self.images_dir.glob("*.png"))
        
        # Группировка по исходному изображению (по префиксу 1-, 2-, 3-)
        grouped = {}
        for f in image_files:
            prefix = f.stem.split('-')[0]  # "1", "2", "3"
            if prefix not in grouped:
                grouped[prefix] = []
            grouped[prefix].append(f)
        
        # Сортировка внутри каждой группы по размеру
        for prefix in grouped:
            grouped[prefix] = sorted(grouped[prefix], key=lambda x: int(x.stem.split('-')[1]))
        
        return grouped
    
    def get_image_size(self, img_path):
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
            if result.boxes is not None:
                num = len(result.boxes)
                avg_conf = result.boxes.conf.mean().item() if num > 0 else 0
            else:
                num = 0
                avg_conf = 0
            
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
        
        for prefix, images in grouped_images.items():
            print(f"\n{'='*50}")
            print(f"Исходное изображение {prefix}")
            print(f"{'='*50}")
            
            for img_path in tqdm(images, desc=f"  Обработка"):
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
                
                print(f"    {img_name}: {width}×{height} | "
                      f"Время: {result['avg_time_ms']:.1f}±{result['std_time_ms']:.1f} мс | "
                      f"Детекций: {result['avg_detections']:.0f} | "
                      f"Уверенность: {result['avg_confidence']:.3f}")
    
    def save_results_csv(self, output_path="yolo12_size_results_v2.csv"):
        df = pd.DataFrame(self.results)
        df = df.sort_values(['image_prefix', 'height'])
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nРезультаты сохранены в {output_path}")
        return df


class ResultsPlotter:
    """Построение графиков по результатам"""
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
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
        
        print(f"Уникальных размеров: {len(self.df_grouped)}")
    
    def plot_time_vs_height(self):
        """График: Время обработки от высоты"""
        plt.figure(figsize=(12, 7))
        
        colors = {'1': 'blue', '2': 'green', '3': 'red'}
        markers = {'1': 'o', '2': 's', '3': '^'}
        
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                x = data['height'].values
                y = data['avg_time_ms'].values
                y_err = data['std_time_ms'].values
                
                plt.errorbar(x, y, yerr=y_err, fmt=markers[prefix], 
                            color=colors[prefix], capsize=5, capthick=1, 
                            markersize=8, label=f'Изображение {prefix}', alpha=0.8)
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Время обработки (мс)', fontsize=12)
        plt.title('Зависимость времени обработки от высоты изображения', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xscale('log', base=2)
        plt.xticks([16, 32, 64, 128, 256, 512, 1024, 2048], 
                   ['16', '32', '64', '128', '256', '512', '1024', '2048'])
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'plot_time_vs_height_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_time_vs_height_v2.png")
    
    def plot_confidence_vs_height(self):
        """График: Уверенность от высоты"""
        plt.figure(figsize=(12, 7))
        
        colors = {'1': 'blue', '2': 'green', '3': 'red'}
        markers = {'1': 'o', '2': 's', '3': '^'}
        
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                x = data['height'].values
                y = data['avg_confidence'].values
                y_err = data['std_confidence'].values
                
                plt.errorbar(x, y, yerr=y_err, fmt=markers[prefix], 
                            color=colors[prefix], capsize=5, capthick=1, 
                            markersize=8, label=f'Изображение {prefix}', alpha=0.8)
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Средняя уверенность', fontsize=12)
        plt.title('Зависимость уверенности детекции от высоты изображения', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xscale('log', base=2)
        plt.xticks([16, 32, 64, 128, 256, 512, 1024, 2048], 
                   ['16', '32', '64', '128', '256', '512', '1024', '2048'])
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'plot_confidence_vs_height_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_confidence_vs_height_v2.png")
    
    def plot_detections_vs_height(self):
        """График: Количество детекций от высоты"""
        plt.figure(figsize=(12, 7))
        
        colors = {'1': 'blue', '2': 'green', '3': 'red'}
        markers = {'1': 'o', '2': 's', '3': '^'}
        
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                x = data['height'].values
                y = data['avg_detections'].values
                
                plt.plot(x, y, markers[prefix]+'-', color=colors[prefix], 
                        markersize=8, linewidth=1.5, label=f'Изображение {prefix}', alpha=0.8)
        
        plt.xlabel('Высота изображения (пиксели)', fontsize=12)
        plt.ylabel('Количество обнаруженных объектов', fontsize=12)
        plt.title('Зависимость количества детекций от высоты изображения', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xscale('log', base=2)
        plt.xticks([16, 32, 64, 128, 256, 512, 1024, 2048], 
                   ['16', '32', '64', '128', '256', '512', '1024', '2048'])
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'plot_detections_vs_height_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_detections_vs_height_v2.png")
    
    def plot_all_metrics(self):
        """Сводный график: все метрики"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = {'1': 'blue', '2': 'green', '3': 'red'}
        markers = {'1': 'o', '2': 's', '3': '^'}
        
        # Время
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                axes[0].plot(data['height'], data['avg_time_ms'], 
                            markers[prefix]+'-', color=colors[prefix], 
                            markersize=6, linewidth=1.5, label=f'Изобр. {prefix}')
        axes[0].set_xlabel('Высота (пиксели)')
        axes[0].set_ylabel('Время (мс)')
        axes[0].set_title('Время обработки')
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Уверенность
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                axes[1].plot(data['height'], data['avg_confidence'], 
                            markers[prefix]+'-', color=colors[prefix], 
                            markersize=6, linewidth=1.5, label=f'Изобр. {prefix}')
        axes[1].set_xlabel('Высота (пиксели)')
        axes[1].set_ylabel('Уверенность')
        axes[1].set_title('Уверенность детекции')
        axes[1].set_xscale('log', base=2)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Детекции
        for prefix in ['1', '2', '3']:
            data = self.df_grouped[self.df_grouped['image_prefix'] == prefix]
            if len(data) > 0:
                axes[2].plot(data['height'], data['avg_detections'], 
                            markers[prefix]+'-', color=colors[prefix], 
                            markersize=6, linewidth=1.5, label=f'Изобр. {prefix}')
        axes[2].set_xlabel('Высота (пиксели)')
        axes[2].set_ylabel('Количество')
        axes[2].set_title('Количество детекций')
        axes[2].set_xscale('log', base=2)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.suptitle('Сравнение метрик в зависимости от высоты изображения', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'plot_all_metrics_v2.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Сохранён: plot_all_metrics_v2.png")
    
    def generate_latex_table(self, output_path="size_research_table_v2.tex"):
        """Генерация LaTeX таблицы"""
        # Усреднение по всем изображениям для каждого размера
        df_summary = self.df_grouped.groupby('height').agg({
            'avg_time_ms': 'mean',
            'std_time_ms': 'mean',
            'avg_confidence': 'mean',
            'avg_detections': 'mean'
        }).reset_index()
        df_summary = df_summary.sort_values('height')
        
        with open(Path(__file__).parent / output_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Зависимость производительности от высоты изображения}\n")
            f.write("\\label{tab:size_research_v2}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Высота (пкс)} & \\textbf{Размер (Мп)} & \\textbf{Время (мс)} & \\textbf{Уверенность} & \\textbf{Детекций} \\\\\n")
            f.write("\\hline\n")
            
            for _, row in df_summary.iterrows():
                height = int(row['height'])
                pixels_mp = height * (height * 2) / 1e6 if height <= 64 else height * (height * 2) / 1e6
                f.write(f"{height} & {pixels_mp:.2e} & ")
                f.write(f"{row['avg_time_ms']:.1f} $\\pm$ {row['std_time_ms']:.1f} & ")
                f.write(f"{row['avg_confidence']:.3f} & ")
                f.write(f"{row['avg_detections']:.0f} \\\\\n")
                f.write("\\hline\n")
            
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX таблица сохранена в {output_path}")
    
    def plot_all(self):
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ ГРАФИКОВ".center(60))
        print("="*60 + "\n")
        
        self.plot_time_vs_height()
        self.plot_confidence_vs_height()
        self.plot_detections_vs_height()
        self.plot_all_metrics()
        
        print("\n" + "="*60)
        print("ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ".center(60))
        print("="*60)


def main():
    # Путь к модели YOLOv12
    MODEL_PATH = Path(__file__).parent / "yolo12" / "runs" / "train" / "weights" / "best.pt"
    
    if not MODEL_PATH.exists():
        alt_paths = [
            Path("C:/sem8/VKR/DIPLOM_SRC/runs/detect/train/weights/best.pt"),
            Path(__file__).parent.parent / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
        for p in alt_paths:
            if p.exists():
                MODEL_PATH = p
                break
    
    if not MODEL_PATH.exists():
        print(f"Ошибка: модель не найдена")
        return
    
    IMAGES_DIR = Path(__file__).parent / "test_diff_sizes_imgs"
    
    if not IMAGES_DIR.exists():
        print(f"Ошибка: папка {IMAGES_DIR} не найдена")
        return
    
    # Запуск исследования
    researcher = YOLOv12SizeResearchV2(
        model_path=str(MODEL_PATH),
        images_dir=str(IMAGES_DIR),
        num_runs=10,
        warmup_runs=5
    )
    
    researcher.run_research()
    df = researcher.save_results_csv()
    
    # Построение графиков
    plotter = ResultsPlotter("yolo12_size_results_v2.csv")
    plotter.plot_all()
    plotter.generate_latex_table()


if __name__ == "__main__":
    main()