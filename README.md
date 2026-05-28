# DIPLOM_SRC — детекция слизней (YOLO + Faster R-CNN)

## Модульная структура

```
DIPLOM_SRC/
├── slug_detection/              # основной Python-пакет
│   ├── paths.py                 # REPO_ROOT, пути к каталогам
│   ├── bootstrap.py             # добавление корня в sys.path
│   ├── config/
│   │   └── dataset.py           # data.yaml, сплиты, веса YOLO
│   └── app/
│       └── detect_slugs.py      # Streamlit UI
├── detectors/                   # обучение и артефакты моделей
│   ├── yolo11/  yolo12/  yolo26/
│   └── faster_rcnn/
├── investigation/               # сравнения моделей, графики
├── scripts/data/
│   └── json_to_yolo.py          # LabelMe JSON → YOLO
├── dataset/                     # данные (в .gitignore)
├── runs/                        # веса Ultralytics (в .gitignore)
├── app/detect_slugs.py          # shim → slug_detection.app
└── dataset_yaml.py              # shim → slug_detection.config
```

## Окружение

**Windows (PowerShell):**

```bash
python -m venv yolo_env
yolo_env\Scripts\activate
pip install ultralytics torch torchvision streamlit opencv-python pillow pyyaml
```

## Запуск

Все команды — **из корня репозитория**.

### Streamlit (инференс)

```bash
python -m streamlit run app/detect_slugs.py
```

или напрямую:

```bash
python -m streamlit run slug_detection/app/detect_slugs.py
```

### Обучение YOLO

```bash
python detectors/yolo11/train_yolo11.py
python detectors/yolo12/train_yolo12.py --model yolo12n.pt
python detectors/yolo26/train_yolo26.py
```

### Faster R-CNN

```bash
python detectors/faster_rcnn/prepare_data_for_faster_rcnn.py
python detectors/faster_rcnn/train_faster_rcnn_improved.py
```

### Утилиты данных

```bash
python scripts/data/json_to_yolo.py
# или: python json-to-yolo.py
```

### Исследования

```bash
python investigation/yolos/compare_yolos.py
python investigation/yolo_frcnn/compare_yolo12_vs_rcnn.py
```

## Импорт в коде

```python
from slug_detection.paths import REPO_ROOT, DETECTORS_DIR, DATASET_DIR
from slug_detection.config import dataset

data_yaml = dataset.find_data_yaml()
```

Старый стиль (`import dataset_yaml`) по-прежнему работает через shim `dataset_yaml.py`.

## Данные

| Путь | Назначение |
|------|------------|
| `dataset/data.yaml` | конфиг YOLO |
| `detectors/*/runs/.../best.pt` | обученные веса |
| `detectors/faster_rcnn/faster_rcnn_best.pth` | веса FRCNN |
| `get_photos_from_archive/` | сырые фото (локально, .gitignore) |
