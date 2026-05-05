import gradio as gr
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import cv2
import shutil

# Загружаем модель (один раз при старте)
model = YOLO("runs/detect/train/weights/best.pt")

class ImageProcessor:
    def __init__(self):
        self.original_images = []      # хранит пути к оригинальным файлам
        self.annotated_images = []     # хранит пути к размеченным файлам
        self.image_names = []          # имена файлов
        self.counts = []               # количество слизней на каждом
        
    def process_folder(self, folder_path):
        """Обрабатывает все изображения в папке"""
        self.clear()
        
        # Поддерживаемые форматы
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Собираем все изображения в папке
        image_files = []
        for ext in extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            return [], [], [], "❌ В выбранной папке нет изображений"
        
        # Обрабатываем каждое изображение
        for img_path in sorted(image_files):
            # Оригинал
            self.original_images.append(str(img_path))
            self.image_names.append(img_path.name)
            
            # Предсказание
            results = model.predict(source=str(img_path), conf=0.25, save=False)
            
            for r in results:
                annotated = r.plot()  # BGR
                
                # Сохраняем во временный файл
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cv2.imwrite(temp_path, annotated)
                self.annotated_images.append(temp_path)
                
                # Количество слизней
                num = len(r.boxes) if r.boxes is not None else 0
                self.counts.append(num)
        
        return self.original_images, self.annotated_images, self.counts, f"✅ Обработано {len(self.original_images)} изображений"
    
    def process_files(self, file_paths):
        """Обрабатывает выбранные файлы (список путей)"""
        self.clear()
        
        if not file_paths:
            return [], [], [], "❌ Не выбрано ни одного файла"
        
        for file_path in file_paths:
            # Получаем оригинальный путь (Gradio передаёт объект с атрибутом 'name')
            if hasattr(file_path, 'name'):
                img_path = file_path.name
            else:
                img_path = str(file_path)
            
            self.original_images.append(img_path)
            self.image_names.append(Path(img_path).name)
            
            # Предсказание
            results = model.predict(source=img_path, conf=0.25, save=False)
            
            for r in results:
                annotated = r.plot()
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cv2.imwrite(temp_path, annotated)
                self.annotated_images.append(temp_path)
                
                num = len(r.boxes) if r.boxes is not None else 0
                self.counts.append(num)
        
        return self.original_images, self.annotated_images, self.counts, f"✅ Обработано {len(self.original_images)} изображений"
    
    def clear(self):
        """Очищает временные файлы и списки"""
        # Удаляем временные размеченные файлы
        for path in self.annotated_images:
            try:
                os.unlink(path)
            except:
                pass
        self.original_images = []
        self.annotated_images = []
        self.image_names = []
        self.counts = []
    
    def get_current_pair(self, index):
        """Возвращает пару (оригинал, разметка) по индексу"""
        if 0 <= index < len(self.original_images):
            return self.original_images[index], self.annotated_images[index], self.counts[index], self.image_names[index]
        return None, None, 0, ""

# Глобальный экземпляр процессора
processor = ImageProcessor()

# Состояние для навигации
current_index_state = gr.State(0)
total_count_state = gr.State(0)

def load_folder(folder_path):
    """Загружает и обрабатывает папку"""
    if not folder_path:
        return [], [], 0, 0, "❌ Выберите папку", gr.update(visible=False), gr.update(visible=False)
    
    orig, annot, counts, msg = processor.process_folder(folder_path)
    total = len(orig)
    
    if total == 0:
        return [], [], 0, 0, msg, gr.update(visible=False), gr.update(visible=False)
    
    # Возвращаем первое изображение для отображения
    if orig:
        return (
            orig[0] if orig else None,
            annot[0] if annot else None,
            counts[0] if counts else 0,
            total,
            msg,
            gr.update(visible=True, value=1),
            gr.update(visible=True)
        )
    return None, None, 0, 0, msg, gr.update(visible=False), gr.update(visible=False)

def load_files(files):
    """Загружает и обрабатывает выбранные файлы"""
    if not files:
        return [], [], 0, 0, "❌ Выберите файлы", gr.update(visible=False), gr.update(visible=False)
    
    orig, annot, counts, msg = processor.process_files(files)
    total = len(orig)
    
    if total == 0:
        return [], [], 0, 0, msg, gr.update(visible=False), gr.update(visible=False)
    
    if orig:
        return (
            orig[0],
            annot[0],
            counts[0],
            total,
            msg,
            gr.update(visible=True, value=1),
            gr.update(visible=True)
        )
    return None, None, 0, 0, msg, gr.update(visible=False), gr.update(visible=False)

def next_image(current_idx, total):
    """Переход к следующему изображению"""
    new_idx = current_idx + 1
    if new_idx >= total:
        new_idx = 0  # зацикливание
    
    orig = processor.original_images[new_idx] if new_idx < len(processor.original_images) else None
    annot = processor.annotated_images[new_idx] if new_idx < len(processor.annotated_images) else None
    count = processor.counts[new_idx] if new_idx < len(processor.counts) else 0
    
    return orig, annot, count, new_idx + 1, new_idx

def prev_image(current_idx, total):
    """Переход к предыдущему изображению"""
    new_idx = current_idx - 1
    if new_idx < 0:
        new_idx = total - 1
    
    orig = processor.original_images[new_idx] if new_idx < len(processor.original_images) else None
    annot = processor.annotated_images[new_idx] if new_idx < len(processor.annotated_images) else None
    count = processor.counts[new_idx] if new_idx < len(processor.counts) else 0
    
    return orig, annot, count, new_idx + 1, new_idx

# Создаём интерфейс Gradio
with gr.Blocks(title="YOLO Детектор слизней", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐌 Детектор слизней на изображениях")
    gr.Markdown("Выберите **папку** с изображениями или **отдельные файлы**, затем нажмите **Загрузить и детектировать**")
    
    with gr.Row():
        # Левая колонка: выбор источника
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Источник изображений")
            
            folder_input = gr.Textbox(
                label="Путь к папке",
                placeholder="Например: C:/Users/User/images",
                info="Укажите путь к папке с изображениями"
            )
            folder_btn = gr.Button("📁 Загрузить папку", variant="secondary")
            
            gr.Markdown("--- или ---")
            
            file_input = gr.File(
                file_count="multiple",
                label="Выберите файлы",
                file_types=["image"]
            )
            file_btn = gr.Button("🖼️ Загрузить файлы", variant="secondary")
        
        # Правая колонка: статус
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Статус", interactive=False)
    
    gr.Markdown("---")
    gr.Markdown("### 🖱️ Результаты детекции")
    
    # Два столбца для изображений: исходное и размеченное
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 📷 Исходное изображение")
            original_image = gr.Image(label="Оригинал", type="filepath", height=400)
            
        with gr.Column(scale=1):
            gr.Markdown("#### 🐍 Размеченное изображение")
            annotated_image = gr.Image(label="Детекция", type="filepath", height=400)
    
    # Счётчик и кнопки навигации
    with gr.Row():
        prev_btn = gr.Button("◀ Назад", variant="secondary", visible=False)
        counter_text = gr.Markdown("**Изображение: 0 / 0**", visible=False)
        next_btn = gr.Button("Вперёд ▶", variant="secondary", visible=False)
    
    with gr.Row():
        slug_count = gr.Markdown("**🐌 Количество слизней:** --", visible=False)
    
    # Скрытые состояния
    current_idx = gr.State(0)
    total_images = gr.State(0)
    
    # Обработчики событий
    folder_btn.click(
        fn=load_folder,
        inputs=[folder_input],
        outputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn]
    )
    
    file_btn.click(
        fn=load_files,
        inputs=[file_input],
        outputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn]
    )
    
    # Обновляем markdown со счётчиком и количеством
    def update_display(orig, annot, count, idx, total):
        # Обновляем slug_count
        slug_md = f"**🐌 Количество слизней:** {count}" if count is not None else "**🐌 Количество слизней:** --"
        # Обновляем counter_text
        counter_md = f"**Изображение:** {idx} / {total}" if total > 0 else "**Изображение:** 0 / 0"
        return orig, annot, slug_md, counter_md
    
    # Функция для next с обновлением дисплея
    def next_with_update(current_idx, total):
        orig, annot, count, new_idx_display, new_idx = next_image(current_idx, total)
        slug_md = f"**🐌 Количество слизней:** {count}" if count is not None else "**🐌 Количество слизней:** --"
        counter_md = f"**Изображение:** {new_idx_display} / {total}" if total > 0 else "**Изображение:** 0 / 0"
        return orig, annot, slug_md, counter_md, new_idx
    
    def prev_with_update(current_idx, total):
        orig, annot, count, new_idx_display, new_idx = prev_image(current_idx, total)
        slug_md = f"**🐌 Количество слизней:** {count}" if count is not None else "**🐌 Количество слизней:** --"
        counter_md = f"**Изображение:** {new_idx_display} / {total}" if total > 0 else "**Изображение:** 0 / 0"
        return orig, annot, slug_md, counter_md, new_idx
    
    # Привязываем навигацию
    next_btn.click(
        fn=next_with_update,
        inputs=[current_idx, total_images],
        outputs=[original_image, annotated_image, slug_count, counter_text, current_idx]
    )
    
    prev_btn.click(
        fn=prev_with_update,
        inputs=[current_idx, total_images],
        outputs=[original_image, annotated_image, slug_count, counter_text, current_idx]
    )
    
    # При загрузке данных обновляем current_idx и total_images
    def on_load(orig, annot, count, total, msg, counter, prev, nxt):
        # Возвращаем всё то же самое, плюс устанавливаем current_idx = 0, total_images = total
        return orig, annot, count, total, msg, counter, prev, nxt, 0, total
    
    folder_btn.click(
        fn=on_load,
        inputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn],
        outputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn, current_idx, total_images]
    )
    
    file_btn.click(
        fn=on_load,
        inputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn],
        outputs=[original_image, annotated_image, slug_count, total_images, status_text, counter_text, prev_btn, next_btn, current_idx, total_images]
    )
    
    gr.Markdown("---\n💡 **Совет:** Для изменения порога уверенности отредактируйте параметр `conf=0.25` в коде")

if __name__ == "__main__":
    demo.launch(inbrowser=True)