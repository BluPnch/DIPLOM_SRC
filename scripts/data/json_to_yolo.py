"""Конвертация LabelMe JSON → YOLO txt (класс slug)."""

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.paths import DATASET_DIR, REPO_ROOT

INPUT_JSON_DIR = DATASET_DIR / "labels" / "filtered"
OUTPUT_LABELS_DIR = DATASET_DIR / "labels" / "test"


def main() -> None:
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    if not INPUT_JSON_DIR.is_dir():
        raise FileNotFoundError(
            f"Не найден каталог с JSON: {INPUT_JSON_DIR}\n"
            f"Корень репозитория: {REPO_ROOT}"
        )

    for file in os.listdir(INPUT_JSON_DIR):
        if not file.endswith(".json"):
            continue

        json_path = INPUT_JSON_DIR / file

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        yolo_lines = []

        for shape in data["shapes"]:
            if shape["label"] != "slug":
                continue

            (x1, y1), (x2, y2) = shape["points"]

            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h

            yolo_lines.append(f"0 {x_center} {y_center} {width} {height}")

        txt_name = os.path.splitext(file)[0] + ".txt"
        txt_path = OUTPUT_LABELS_DIR / txt_name

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

    print("Конвертация завершена.")


if __name__ == "__main__":
    main()
