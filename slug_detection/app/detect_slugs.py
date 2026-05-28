"""Streamlit UI: детекция слизней (YOLOv11/12/26 и Faster R-CNN)."""

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

from slug_detection.paths import DETECTORS_DIR, REPO_ROOT

st.set_page_config(page_title="Detector Slugs", layout="wide")
st.title("Slug Detector on Images")


class ModelLoader:
    """Model loader with caching"""

    def __init__(self):
        self.project_root = REPO_ROOT
        self.detectors_dir = DETECTORS_DIR

        self.yolo26_path = self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt"
        self.yolo12_path = self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt"
        self.yolo11_path = self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt"
        self.frcnn_path = self.detectors_dir / "faster_rcnn" / "faster_rcnn_best.pth"

        self.yolo12_alternatives = [
            self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train2" / "weights" / "best.pt",
        ]

        self.yolo11_alternatives = [
            self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]

        self.yolo26_alternatives = [
            self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt",
            self.detectors_dir / "yolo" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]

        self.frcnn_model = None

    def find_yolo12_weights(self):
        for path in self.yolo12_alternatives:
            if path.exists():
                return path
        return None

    def find_yolo11_weights(self):
        for path in self.yolo11_alternatives:
            if path.exists():
                return path
        return None

    def find_yolo26_weights(self):
        for path in self.yolo26_alternatives:
            if path.exists():
                return path
        return None

    def load_yolo26(self):
        yolo26_path = self.find_yolo26_weights()
        if yolo26_path is not None:
            return YOLO(str(yolo26_path))
        st.warning("Trained YOLOv26 model not found, using pretrained yolo26n.pt")
        return YOLO("yolo26n.pt")

    def load_yolo12(self):
        yolo12_path = self.find_yolo12_weights()
        if yolo12_path is not None:
            return YOLO(str(yolo12_path))
        st.warning("Trained YOLOv12 model not found, using pretrained yolo12n.pt")
        return YOLO("yolo12n.pt")

    def load_yolo11(self):
        yolo11_path = self.find_yolo11_weights()
        if yolo11_path is not None:
            return YOLO(str(yolo11_path))
        st.warning("Trained YOLOv11 model not found, using pretrained yolo11n.pt")
        return YOLO("yolo11n.pt")

    def load_faster_rcnn(self):
        if not self.frcnn_path.exists():
            st.error(f"Faster R-CNN model not found: {self.frcnn_path}")
            return None

        try:
            import torchvision
            from torchvision.models.detection import FasterRCNN
            from torchvision.models.detection.rpn import AnchorGenerator

            # Должно совпадать с detectors/faster_rcnn/train_faster_rcnn_improved.py
            backbone = torchvision.models.mobilenet_v3_small(weights="DEFAULT").features
            backbone.out_channels = 576

            anchor_generator = AnchorGenerator(
                sizes=((16, 32, 64, 128, 256),),
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
                min_size=640,
                max_size=640,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                state = torch.load(
                    str(self.frcnn_path), map_location=device, weights_only=True
                )
            except TypeError:
                state = torch.load(str(self.frcnn_path), map_location=device)
            model.load_state_dict(state)
            model.eval()
            model.to(device)

            return model
        except Exception as e:
            st.error(f"Faster R-CNN load error: {e}")
            return None


@st.cache_resource
def load_model(model_type="YOLOv12"):
    loader = ModelLoader()

    if model_type == "YOLOv11":
        return loader.load_yolo11(), "yolo"
    if model_type == "YOLOv12":
        return loader.load_yolo12(), "yolo"
    if model_type == "YOLOv26":
        return loader.load_yolo26(), "yolo"
    return loader.load_faster_rcnn(), "faster_rcnn"


def predict_faster_rcnn(model, image_path, conf_threshold, temperature):
    from torchvision import transforms

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image = image.resize((640, 640), Image.Resampling.BILINEAR)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    pred = predictions[0]

    scores = torch.sigmoid(pred["scores"] / temperature).cpu().numpy()

    keep = scores > conf_threshold
    boxes = pred["boxes"][keep].cpu().numpy()
    scores = scores[keep]

    if len(boxes) > 0:
        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    return boxes, scores


BOX_LINE_THICKNESS = 4
LABEL_FONT_SCALE = 1.1
LABEL_FONT_THICKNESS = 3


def draw_boxes_on_image(
    image_path: str,
    boxes_xyxy,
    scores,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    label_prefix: str = "slug",
    line_thickness: int = BOX_LINE_THICKNESS,
    font_scale: float = LABEL_FONT_SCALE,
    font_thickness: int = LABEL_FONT_THICKNESS,
) -> "cv2.Mat":
    """Рисует рамки поверх оригинала (BGR), без перекрашивания всего кадра."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        label = f"{label_prefix}: {float(score):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_y = max(y1 - 8, text_h + baseline)
        text_x = x1

        cv2.putText(
            img,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )
    return img


def process_images(image_paths, model, model_type, conf_threshold, iou_threshold, temperature=1.8):
    st.session_state.image_paths = []
    st.session_state.annotated_paths = []
    st.session_state.counts = []
    st.session_state.filenames = []
    st.session_state.current_index = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, img_path in enumerate(image_paths):
        status_text.text(f"Processing {i+1}/{len(image_paths)}: {Path(img_path).name}")

        st.session_state.image_paths.append(img_path)
        st.session_state.filenames.append(Path(img_path).name)

        if model_type == "yolo":
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                num = len(boxes_xyxy)
            else:
                boxes_xyxy = []
                scores = []
                num = 0
            annotated = draw_boxes_on_image(img_path, boxes_xyxy, scores)
        else:
            boxes, scores = predict_faster_rcnn(model, img_path, conf_threshold, temperature)
            num = len(boxes)
            annotated = draw_boxes_on_image(img_path, boxes, scores)

        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, annotated)
        st.session_state.annotated_paths.append(temp_path)
        st.session_state.counts.append(num)

        progress_bar.progress((i + 1) / len(image_paths))

    status_text.text(f"Processed {len(image_paths)} images")
    return len(image_paths) > 0


def load_from_files(files, model, model_type, conf_threshold, iou_threshold, temperature):
    if not files:
        return False

    temp_dir = tempfile.mkdtemp()
    image_paths = []

    for file in files:
        img_path = os.path.join(temp_dir, file.name)
        with open(img_path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(img_path)

    return process_images(
        image_paths, model, model_type, conf_threshold, iou_threshold, temperature
    )


def next_image():
    if st.session_state.current_index < len(st.session_state.image_paths) - 1:
        st.session_state.current_index += 1


def prev_image():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1


def _init_session_state() -> None:
    defaults = {
        "current_index": 0,
        "image_paths": [],
        "annotated_paths": [],
        "counts": [],
        "filenames": [],
        "conf_threshold": 0.3,
        "iou_threshold": 0.5,
        "model_choice": "YOLOv12",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


with st.sidebar:
    st.header("Model Selection")

    model_choice = st.radio(
        "Select model for detection:",
        ["YOLOv11", "YOLOv12", "YOLOv26", "Faster R-CNN"],
        key="model_choice",
    )

    model, model_type = load_model(st.session_state.model_choice)

    if model is None:
        st.error(f"Failed to load model: {st.session_state.model_choice}")
        st.stop()

    st.markdown("---")

    st.header("Detection Parameters")

    st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(st.session_state.conf_threshold),
        step=0.05,
        key="conf_threshold",
    )

    st.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=float(st.session_state.iou_threshold),
        step=0.05,
        key="iou_threshold",
    )

    conf_threshold = st.session_state.conf_threshold
    iou_threshold = st.session_state.iou_threshold
    temperature = 1.8  # фиксировано для Faster R-CNN, для YOLO не используется

    st.markdown("---")
    st.header("Image Upload")

    uploaded_files = st.file_uploader(
        "Select images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption(f"Selected: {len(uploaded_files)} file(s). Click **Run Detection** to process.")

    if st.button("Run Detection", type="primary"):
        if uploaded_files:
            with st.spinner(f"Processing with {model_choice}..."):
                try:
                    ok = load_from_files(
                        uploaded_files,
                        model,
                        model_type,
                        conf_threshold,
                        iou_threshold,
                        temperature,
                    )
                except Exception as exc:
                    st.error(f"Detection error: {exc}")
                    ok = False
                if ok:
                    st.success(f"Loaded {len(st.session_state.image_paths)} images")
                    st.rerun()
                else:
                    st.error("Failed to load images")
        else:
            st.error("Select at least one image")

    st.markdown("---")

    if st.session_state.image_paths:
        st.markdown("### Statistics")
        st.markdown(f"**Total images:** {len(st.session_state.image_paths)}")
        st.markdown(f"**Total slugs:** {sum(st.session_state.counts)}")
        st.markdown(
            f"**Current:** {st.session_state.current_index + 1} / {len(st.session_state.image_paths)}"
        )

        st.markdown("---")
        st.markdown(f"**Model:** {model_choice}")
        st.markdown(f"**Confidence threshold:** {conf_threshold:.2f}")
        st.markdown(f"**IoU threshold:** {iou_threshold:.2f}")

        if len(st.session_state.counts) > 0:
            st.markdown(
                f"**Average slugs per image:** {sum(st.session_state.counts) / len(st.session_state.counts):.1f}"
            )


conf_threshold = st.session_state.conf_threshold
iou_threshold = st.session_state.iou_threshold

if st.session_state.image_paths:
    idx = st.session_state.current_index

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("First", use_container_width=True):
            st.session_state.current_index = 0
            st.rerun()
    with col2:
        if st.button("Prev", use_container_width=True):
            prev_image()
            st.rerun()
    with col4:
        if st.button("Next", use_container_width=True):
            next_image()
            st.rerun()
    with col5:
        if st.button("Last", use_container_width=True):
            st.session_state.current_index = len(st.session_state.image_paths) - 1
            st.rerun()

    st.caption(f"Confidence: {conf_threshold:.2f} | IoU: {iou_threshold:.2f}")
    st.progress((idx + 1) / len(st.session_state.image_paths))

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Original Image")
        st.image(st.session_state.image_paths[idx], use_container_width=True)

    with col_right:
        st.markdown("### Detection Result")
        st.image(st.session_state.annotated_paths[idx], use_container_width=True)

    st.markdown("---")
    st.markdown(f"## Slugs detected: **{st.session_state.counts[idx]}**")
    st.markdown("---")
    st.markdown(
        f"**File:** {st.session_state.filenames[idx]} ({idx + 1} / {len(st.session_state.image_paths)})"
    )

else:
    st.info("Select a model, adjust parameters, then upload images in the sidebar")
