import os
import json

input_json_dir = "dataset\\labels\\filtered"
output_labels_dir = "dataset\\labels\\test"

os.makedirs(output_labels_dir, exist_ok=True)

for file in os.listdir(input_json_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(input_json_dir, file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        if shape["label"] != "slug":
            continue

        (x1, y1), (x2, y2) = shape["points"]

        # координаты
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h

        yolo_lines.append(f"0 {x_center} {y_center} {width} {height}")

    txt_name = os.path.splitext(file)[0] + ".txt"
    txt_path = os.path.join(output_labels_dir, txt_name)

    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

print("Конвертация завершена.")
