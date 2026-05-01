import gradio as gr
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path

# Загружаем модель (один раз при старте)
model = YOLO("runs/detect/train/weights/best.pt")

def predict_and_display(images):
    """
    images: список загруженных файлов (пути к временным копиям)
    Возвращает: галерею изображений с разметкой и подписи (имя файла + кол-во объектов)
    """
    if not images:
        return [], "Не выбрано ни одного изображения"

    results_list = []      # для хранения путей к размеченным картинкам
    captions = []           # для подписей под каждым изображением
    
    for img_path in images:
        # Предсказание
        results = model.predict(source=img_path, conf=0.25, save=False)
        
        for r in results:
            # Получаем изображение с боксами (numpy-массив)
            annotated = r.plot()   # BGR (OpenCV)
            # Сохраняем во временный файл, чтобы Gradio мог его показать
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            # r.plot() возвращает BGR, а OpenCV сохраняет в BGR – корректно
            import cv2
            cv2.imwrite(temp_path, annotated)
            results_list.append(temp_path)
            
            # Количество обнаруженных объектов
            num = len(r.boxes) if r.boxes is not None else 0
            caption = f"{Path(img_path).name} | объектов: {num}"
            captions.append(caption)
    
    return results_list, "\n".join(captions)

# Создаём интерфейс Gradio
with gr.Blocks(title="YOLO Детектор слизней") as demo:
    gr.Markdown("# 🐌 Детектор слизней на фото")
    gr.Markdown("Выберите одно или несколько изображений, нажмите **Предсказать** и листайте результаты с помощью стрелок.")
    
    with gr.Row():
        input_files = gr.File(
            file_count="multiple",          # разрешаем выбрать несколько файлов
            label="Выберите изображения",
            file_types=["image"]
        )
        predict_btn = gr.Button("🔍 Предсказать", variant="primary")
    
    with gr.Row():
        # Галерея с навигацией (стрелки)
        gallery = gr.Gallery(
            label="Результаты детекции",
            show_label=True,
            columns=1,                     # показываем по одному изображению за раз
            rows=1,
            object_fit="contain",
            height="auto",
            allow_preview=False,
            interactive=False
        )
    
    caption_output = gr.Textbox(label="Информация по каждому изображению", lines=5)
    
    # При нажатии кнопки запускаем обработку
    predict_btn.click(
        fn=predict_and_display,
        inputs=[input_files],
        outputs=[gallery, caption_output]
    )
    
    gr.Markdown("---\n💡 **Совет:** Увеличьте порог уверенности (conf) в коде, если много ложных срабатываний.")

if __name__ == "__main__":
    demo.launch()