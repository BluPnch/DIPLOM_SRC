import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dataset_yaml  # noqa: E402


class ModelComparator:
    """Сравнение производительности моделей"""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.frcnn_dir = Path(__file__).resolve().parent
        self.last_yolo_weights: Path | None = None

    def load_yolo(self, model_path=None):
        if model_path is not None:
            p = Path(model_path)
            if not p.is_file():
                raise FileNotFoundError(f"Не найден файл весов YOLO: {p}")
            self.last_yolo_weights = p.resolve()
            return YOLO(str(self.last_yolo_weights))

        resolved = dataset_yaml.resolve_yolo_best_weights(self.project_root)
        if resolved is None:
            raise FileNotFoundError(
                "Не найден ни один best.pt (искали под runs/, detectors/yolo/runs/…).\n"
                "Обучите: python detectors/yolo/train_yolo.py\n"
                "Или укажите путь: load_yolo(r'C:\\path\\to\\best.pt')."
            )
        self.last_yolo_weights = resolved.resolve()
        default_pt = (
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
        )
        if self.last_yolo_weights != default_pt.resolve():
            print(f"Использую веса YOLO: {self.last_yolo_weights}")
        return YOLO(str(self.last_yolo_weights))

    def load_faster_rcnn(self, model_path=None):
        if model_path is None:
            model_path = self.frcnn_dir / "faster_rcnn_best.pth"
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Нет Faster R-CNN весов: {model_path}\n"
                "Сначала: python detectors/faster_rcnn/train_faster_rcnn.py"
            )

        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator

        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2,
        )

        model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        model.load_state_dict(torch.load(str(model_path)))
        model.eval()
        return model

    def measure_fps(self, model, model_type, num_runs=100):
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for _ in range(10):
            if model_type == "yolo":
                model(test_img, verbose=False)
            else:
                from torchvision import transforms

                img_tensor = transforms.ToTensor()(test_img).unsqueeze(0)
                with torch.no_grad():
                    model(img_tensor)

        times = []
        for _ in range(num_runs):
            start = time.time()

            if model_type == "yolo":
                model(test_img, verbose=False)
            else:
                from torchvision import transforms

                img_tensor = transforms.ToTensor()(test_img).unsqueeze(0)
                with torch.no_grad():
                    model(img_tensor)

            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        return fps, avg_time

    def get_model_size(self, model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)

    def evaluate_accuracy(self, yolo_model, faster_model, val_images_dir, val_labels_dir):
        val_images_dir = Path(val_images_dir)
        val_labels_dir = Path(val_labels_dir)

        yolo_metrics = yolo_model.val()

        faster_metrics = self.evaluate_faster_rcnn(
            faster_model, val_images_dir, val_labels_dir
        )

        return {
            "yolo": {
                "mAP50": yolo_metrics.box.map50,
                "mAP50_95": yolo_metrics.box.map,
                "precision": yolo_metrics.box.mp,
                "recall": yolo_metrics.box.mr,
            },
            "faster_rcnn": faster_metrics,
        }

    def load_yolo_annotations(self, label_path, img_size):
        """GT в формате YOLO (норм. xywh) → dict для torchmetrics (xyxy, пиксели)."""
        w_img, h_img = img_size
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x_min = (xc - bw / 2) * w_img
                        y_min = (yc - bh / 2) * h_img
                        x_max = (xc + bw / 2) * w_img
                        y_max = (yc + bh / 2) * h_img
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(cls_id + 1)

        if not boxes:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def evaluate_faster_rcnn(self, model, val_images_dir, val_labels_dir):
        from torchmetrics.detection import MeanAveragePrecision
        from torchvision import transforms

        metric = MeanAveragePrecision()

        val_images_dir = Path(val_images_dir)
        for img_path in dataset_yaml.iter_image_paths(val_images_dir):
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)

            with torch.no_grad():
                preds = model(img_tensor)

            label_path = val_labels_dir / f"{img_path.stem}.txt"
            targets = self.load_yolo_annotations(label_path, img.size)

            metric.update(preds, [targets])

        return metric.compute()

    def create_comparison_table(self):
        print("\n" + "=" * 60)
        print("Сравнение эффективности детекторов".center(60))
        print("=" * 60)

        print("Загрузка моделей...")
        yolo = self.load_yolo()
        faster = self.load_faster_rcnn()

        print("Измерение FPS...")
        yolo_fps, yolo_time = self.measure_fps(yolo, "yolo")
        faster_fps, faster_time = self.measure_fps(faster, "faster_rcnn")

        print("Оценка точности...")
        data_cfg, yaml_path = dataset_yaml.load_data_cfg()
        val_images = dataset_yaml.resolve_split_images_dir(data_cfg, "val", yaml_path)
        val_labels = dataset_yaml.yolo_labels_dir(val_images, "val")
        accuracy = self.evaluate_accuracy(yolo, faster, val_images, val_labels)

        print("Размер моделей...")
        if self.last_yolo_weights is None:
            raise RuntimeError("Не задан путь к весам YOLO после load_yolo.")
        yolo_size = self.get_model_size(str(self.last_yolo_weights))
        faster_size = self.get_model_size(str(self.frcnn_dir / "faster_rcnn_best.pth"))

        fm_raw = accuracy["faster_rcnn"]
        fm = {
            k: (v.item() if torch.is_tensor(v) else float(v))
            for k, v in fm_raw.items()
        }

        table_data = {
            "Метрика": [
                "mAP@0.5 (%)",
                "mAP@0.5:0.95 (%)",
                "Precision (%)",
                "Recall (%)",
                "FPS (кадр/сек)",
                "Время инференса (мс)",
                "Размер модели (MB)",
            ],
            "YOLO": [
                f"{accuracy['yolo']['mAP50']:.2f}",
                f"{accuracy['yolo']['mAP50_95']:.2f}",
                f"{accuracy['yolo']['precision']:.2f}",
                f"{accuracy['yolo']['recall']:.2f}",
                f"{yolo_fps:.1f}",
                f"{yolo_time*1000:.1f}",
                f"{yolo_size:.1f}",
            ],
            "Faster R-CNN": [
                f"{fm.get('map_50', fm.get('map', 0.0)):.2f}",
                f"{fm.get('map', 0.0):.2f}",
                f"{fm.get('map', 0.0):.2f}",
                f"{fm.get('mar_100', fm.get('map', 0.0)):.2f}",
                f"{faster_fps:.1f}",
                f"{faster_time*1000:.1f}",
                f"{faster_size:.1f}",
            ],
        }

        print("\n" + "-" * 80)
        print(f"{'Метрика':<25} {'YOLO':<25} {'Faster R-CNN':<25}")
        print("-" * 80)
        for i in range(len(table_data["Метрика"])):
            print(
                f"{table_data['Метрика'][i]:<25} {table_data['YOLO'][i]:<25} {table_data['Faster R-CNN'][i]:<25}"
            )
        print("-" * 80)

        return table_data

    def plot_comparison(self, table_data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        metrics = ["mAP@0.5", "Precision", "Recall"]
        yolo_vals = [
            float(table_data["YOLO"][0]),
            float(table_data["YOLO"][2]),
            float(table_data["YOLO"][3]),
        ]
        faster_vals = [
            float(table_data["Faster R-CNN"][0]),
            float(table_data["Faster R-CNN"][2]),
            float(table_data["Faster R-CNN"][3]),
        ]

        x = np.arange(len(metrics))
        width = 0.35

        axes[0].bar(x - width / 2, yolo_vals, width, label="YOLO", color="blue", alpha=0.7)
        axes[0].bar(
            x + width / 2, faster_vals, width, label="Faster R-CNN", color="red", alpha=0.7
        )
        axes[0].set_ylabel("Значение (%)")
        axes[0].set_title("Сравнение точности")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        metrics2 = ["FPS", "Размер (MB)"]
        yolo_vals2 = [float(table_data["YOLO"][4]), float(table_data["YOLO"][6])]
        faster_vals2 = [
            float(table_data["Faster R-CNN"][4]),
            float(table_data["Faster R-CNN"][6]),
        ]

        ymax = max(yolo_vals2[0], faster_vals2[0])
        yolo_vals2[0] = yolo_vals2[0] / ymax * 100
        faster_vals2[0] = faster_vals2[0] / ymax * 100

        x2 = np.arange(len(metrics2))
        axes[1].bar(x2 - width / 2, yolo_vals2, width, label="YOLO", color="blue", alpha=0.7)
        axes[1].bar(
            x2 + width / 2, faster_vals2, width, label="Faster R-CNN", color="red", alpha=0.7
        )
        axes[1].set_ylabel("Нормированное значение (%)")
        axes[1].set_title("Сравнение скорости и размера")
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels(metrics2)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = self.frcnn_dir / "model_comparison.png"
        plt.savefig(str(out_png), dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\n✅ Графики сохранены в '{out_png}'")


if __name__ == "__main__":
    comparator = ModelComparator()
    results = comparator.create_comparison_table()
    comparator.plot_comparison(results)
