import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataset_yaml  # noqa: E402 — после добавления корня проекта в sys.path

FASTER_RCNN_DIR = Path(__file__).resolve().parent
DEFAULT_COCO_DIR = FASTER_RCNN_DIR / "coco_dataset"


def convert_yolo_to_coco(data_yaml_path, output_dir=None):
    """Конвертирует YOLO датасет в COCO формат для Faster R-CNN."""

    if output_dir is None:
        output_dir = DEFAULT_COCO_DIR
    output_dir = Path(output_dir)

    data_yaml_path = Path(data_yaml_path).resolve()

    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Не найден data.yaml: {data_yaml_path}")

    with open(data_yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    classes = dataset_yaml.class_names_from_cfg(data_cfg)
    total_images = 0

    for split in ["train", "val"]:
        images_dir = dataset_yaml.resolve_split_images_dir(data_cfg, split, data_yaml_path)
        labels_dir = dataset_yaml.yolo_labels_dir(images_dir, split)

        if not images_dir.is_dir():
            print(
                f"⚠ {split}: нет папки с изображениями: {images_dir}",
                file=sys.stderr,
            )

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name, "supercategory": "none"}
                for i, name in enumerate(classes, 1)
            ],
        }

        annotation_id = 1
        image_id = 1

        image_paths = dataset_yaml.iter_image_paths(images_dir)
        for img_path in image_paths:
            shutil.copy(img_path, output_dir / split / "images" / img_path.name)

            img = Image.open(img_path)
            width, height = img.size

            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                }
            )

            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0]) + 1
                            x_center, y_center, w, h = map(float, parts[1:5])

                            x_min = (x_center - w / 2) * width
                            y_min = (y_center - h / 2) * height
                            box_width = w * width
                            box_height = h * height

                            coco_data["annotations"].append(
                                {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": class_id,
                                    "bbox": [x_min, y_min, box_width, box_height],
                                    "area": box_width * box_height,
                                    "iscrowd": 0,
                                }
                            )
                            annotation_id += 1

            image_id += 1

        split_n = len(image_paths)
        total_images += split_n
        out_json = output_dir / split / "_annotations.coco.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)

        print(f"✓ {split}: {split_n} изображений, {len(coco_data['annotations'])} аннотаций")

    print(f"✅ Данные сконвертированы в {output_dir}")
    print(f"(исходный YAML: {data_yaml_path})")
    if total_images == 0:
        print(
            "\nНе найдено ни одного изображения (.jpg/.jpeg/.png/…).\n"
            "Проверьте ключи train/val и path в YAML и соответствие папка images / labels.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return output_dir


if __name__ == "__main__":
    convert_yolo_to_coco(dataset_yaml.find_data_yaml())
