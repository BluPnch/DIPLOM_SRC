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

**Linux / WSL (Ubuntu):** команда интерпретатора обычно `python3`, активация — через `bin/`, не `Scripts\`:

```bash
sudo apt install -y python3 python3-venv python3-pip  
python3 -m venv yolo_env
source yolo_env/bin/activate
pip install ultralytics torch torchvision streamlit opencv-python pillow pyyaml
```

**WSL / Linux — если что‑то «не получается»:**

1. **Проверьте активацию** (ваш индикатор в приглашении может не показывать venv):

   ```bash
   echo "$VIRTUAL_ENV"
   which python
   ```

   После активации `which python` должно быть вроде `.../DIPLOM_SRC/yolo_env/bin/python` (или путь ниже для venv в `$HOME`), а переменная `VIRTUAL_ENV` не должна быть пустой.

2. **Не создавайте venv на диске Windows (`/mnt/c/...`), если возможны странные ошибки.** Создайте окружение в домашнем каталоге Linux и активируйте его при работе в проекте:

   ```bash
   mkdir -p ~/venvs
   python3 -m venv ~/venvs/yolo_diplom
   source ~/venvs/yolo_diplom/bin/activate
   cd /mnt/c/sem8/VKR/DIPLOM_SRC
   pip install ultralytics torch torchvision streamlit opencv-python pillow pyyaml
   ```

Обучение YOLO (из корня репозитория):

```bash
python detectors/yolo/train_yolo.py
```

Скрипт сам ставит `device=cpu` и отключает AMP, если CUDA недоступна (как у сборки `torch+cpu`). Веса `best.pt` ищутся в `runs/detect/...`, в `detectors/yolo/runs/...` и в других типичных каталогах Ultralytics.

Перед запуском `train` конфиг автоматически дублируется с **абсолютным `path:`**, чтобы пути из `data.yaml` не смешивались с каталогом `datasets` из `C:\\Users\\...\\Roaming\\Ultralytics\\settings.json`. Если там в пути когда‑то оказался пробел (например `DIPLOM SRC`), исправьте строку на реальный каталог **`...\\DIPLOM_SRC`**.

В `detectors/yolo/dataset/data.yaml` поле `path:` должно указывать на корень датасета (папку с подкаталогом `images/`), например **`path: .`**, если файл лежит уже в этом корне. Не дублируйте имя папки вроде `./dataset`, если конфиг уже лежит внутри `dataset/`.

Streamlit (из корня репозитория):

```bash
python -m streamlit run app/detect_slugs.py
```

Faster R-CNN: сначала конвертация из YOLO в COCO, затем обучение (команды — **из корня** `DIPLOM_SRC`: `cd C:\sem8\VKR\DIPLOM_SRC`). Если текущая папка уже `detectors`, используйте `python faster_rcnn/...` без префикса `detectors/`.

```bash
python detectors/faster_rcnn/prepare_data_for_faster_rcnn.py
python detectors/faster_rcnn/train_faster_rcnn.py
python detectors/faster_rcnn/compare_models.py
```

---

Путь к разметке (LabelImg): указывался локально пользователем; при необходимости используйте свой `labelImg.exe` или аналог.
