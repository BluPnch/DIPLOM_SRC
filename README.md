# DIPLOM_SRC — детекция слизней (YOLO + эксперименты Faster R-CNN)

Структура репозитория:

| Путь | Назначение |
|------|------------|
| `detectors/yolo/train_yolo.py` | Обучение YOLO (Ultralytics) |
| `dataset_yaml.py` | Поиск `data.yaml`: сначала `dataset/data.yaml`, иначе `detectors/yolo/dataset/data.yaml` |
| `app/detect_slugs.py` | Streamlit-интерфейс инференса |
| `detectors/faster_rcnn/` | Подготовка COCO, обучение и сравнение Faster R-CNN |
| `dataset/` или `detectors/yolo/dataset/` | Датасет YOLO + `data.yaml` (часть путей может быть в `.gitignore`) |
| `runs/` | Результаты обучения YOLO и веса `best.pt` (в `.gitignore`) |
| `get_photos_from_archive/` | Сырые фото / разметка (локально, в `.gitignore`) |

Виртуальное окружение:

**Windows (PowerShell, cmd):**

```bash
python -m venv yolo_env
yolo_env\Scripts\activate
pip install ultralytics torch torchvision streamlit opencv-python pillow pyyaml
```



Обучение YOLO (из корня репозитория):

```bash
python detectors/yolo/train_yolo.py
```




Faster R-CNN: 

```bash
python detectors/faster_rcnn/prepare_data_for_faster_rcnn.py
python detectors/faster_rcnn/train_faster_rcnn.py
python detectors/faster_rcnn/compare_models.py
```

---

Streamlit (из корня репозитория):

```bash
python -m streamlit run app/detect_slugs.py
```


Путь к разметке (LabelImg): указывался локально пользователем; при необходимости используйте свой `labelImg.exe` или аналог.
