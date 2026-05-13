# app/detect_slugs.py
import os
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset_yaml

st.set_page_config(page_title="Detector Slugs", layout="wide")
st.title("Slug Detector on Images")


class ModelLoader:
    """Model loader with caching"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.detectors_dir = self.project_root / "detectors"
        
        # Paths for different models
        self.yolo26_path = self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt"
        self.yolo12_path = self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt"
        self.yolo11_path = self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt"
        self.frcnn_path = self.detectors_dir / "faster_rcnn" / "faster_rcnn_best.pth"
        
        # Alternative paths for YOLOv12
        self.yolo12_alternatives = [
            self.detectors_dir / "yolo12" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train2" / "weights" / "best.pt",
        ]
        
        # Alternative paths for YOLOv11
        self.yolo11_alternatives = [
            self.detectors_dir / "yolo11" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
        
        # Alternative paths for YOLOv26
        self.yolo26_alternatives = [
            self.detectors_dir / "yolo26" / "runs" / "train" / "weights" / "best.pt",
            self.detectors_dir / "yolo" / "runs" / "train" / "weights" / "best.pt",
            self.project_root / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
        
        self.frcnn_model = None
    
    def find_yolo12_weights(self):
        """Find YOLOv12 weights"""
        for path in self.yolo12_alternatives:
            if path.exists():
                return path
        return None
    
    def find_yolo11_weights(self):
        """Find YOLOv11 weights"""
        for path in self.yolo11_alternatives:
            if path.exists():
                return path
        return None
    
    def find_yolo26_weights(self):
        """Find YOLOv26 weights"""
        for path in self.yolo26_alternatives:
            if path.exists():
                return path
        return None
    
    def load_yolo26(self):
        """Load YOLOv26 model"""
        yolo26_path = self.find_yolo26_weights()
        if yolo26_path is not None:
            return YOLO(str(yolo26_path))
        st.warning("Trained YOLOv26 model not found, using pretrained yolo26n.pt")
        return YOLO("yolo26n.pt")
    
    def load_yolo12(self):
        """Load YOLOv12 model"""
        yolo12_path = self.find_yolo12_weights()
        if yolo12_path is not None:
            return YOLO(str(yolo12_path))
        st.warning("Trained YOLOv12 model not found, using pretrained yolo12n.pt")
        return YOLO("yolo12n.pt")
    
    def load_yolo11(self):
        """Load YOLOv11 model"""
        yolo11_path = self.find_yolo11_weights()
        if yolo11_path is not None:
            return YOLO(str(yolo11_path))
        st.warning("Trained YOLOv11 model not found, using pretrained yolo11n.pt")
        return YOLO("yolo11n.pt")
    
    def load_faster_rcnn(self):
        """Load Faster R-CNN model"""
        if not self.frcnn_path.exists():
            st.error(f"Faster R-CNN model not found: {self.frcnn_path}")
            return None
        
        try:
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
                min_size=640,
                max_size=640
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(str(self.frcnn_path), map_location=device))
            model.eval()
            model.to(device)
            
            return model
        except Exception as e:
            st.error(f"Faster R-CNN load error: {e}")
            return None


@st.cache_resource
def load_model(model_type="YOLOv12"):
    """Load model with caching"""
    loader = ModelLoader()
    
    if model_type == "YOLOv11":
        return loader.load_yolo11(), "yolo"
    elif model_type == "YOLOv12":
        return loader.load_yolo12(), "yolo"
    elif model_type == "YOLOv26":
        return loader.load_yolo26(), "yolo"
    else:
        return loader.load_faster_rcnn(), "faster_rcnn"


def predict_faster_rcnn(model, image_path, conf_threshold=0.5):
    """Prediction with Faster R-CNN with temperature scaling"""
    from torchvision import transforms
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image = image.resize((640, 640), Image.Resampling.BILINEAR)
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    pred = predictions[0]
    
    # Temperature scaling для калибровки уверенности (temperature=1.5-2.0)
    temperature = 1.8
    scores = torch.sigmoid(pred['scores'] / temperature).cpu().numpy()
    
    keep = scores > conf_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = scores[keep]
    
    if len(boxes) > 0:
        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    
    return boxes, scores

def process_images(image_paths, model, model_type):
    """Process images with selected model"""
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
            results = model.predict(source=img_path, conf=0.3, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)
            annotated = results[0].plot()
            num = len(results[0].boxes) if results[0].boxes is not None else 0
        else:
            boxes, scores = predict_faster_rcnn(model, img_path, conf_threshold=0.3)
            num = len(boxes)
            
            img = cv2.imread(img_path)
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'slug: {score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            annotated = img
        
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(temp_path, annotated)
        st.session_state.annotated_paths.append(temp_path)
        st.session_state.counts.append(num)
        
        progress_bar.progress((i + 1) / len(image_paths))
    
    status_text.text(f"Processed {len(image_paths)} images")
    return len(image_paths) > 0


def load_from_files(files, model, model_type):
    """Load selected files"""
    if not files:
        return False
    
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for file in files:
        img_path = os.path.join(temp_dir, file.name)
        with open(img_path, 'wb') as f:
            f.write(file.getbuffer())
        image_paths.append(img_path)
    
    return process_images(image_paths, model, model_type)


def next_image():
    if st.session_state.current_index < len(st.session_state.image_paths) - 1:
        st.session_state.current_index += 1


def prev_image():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1


# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'annotated_paths' not in st.session_state:
    st.session_state.annotated_paths = []
if 'counts' not in st.session_state:
    st.session_state.counts = []
if 'filenames' not in st.session_state:
    st.session_state.filenames = []


# Sidebar
with st.sidebar:
    st.header("Model Selection")
    
    model_choice = st.radio(
        "Select model for detection:",
        ["YOLOv11", "YOLOv12", "YOLOv26", "Faster R-CNN"],
        help="YOLOv11 - balanced accuracy and localization, YOLOv12 - highest accuracy, YOLOv26 - fastest, Faster R-CNN - slower but precise"
    )
    
    model, model_type = load_model(model_choice)
    
    if model is None:
        st.error(f"Failed to load model: {model_choice}")
        st.stop()
    
    if model_choice == "YOLOv11":
        st.info("YOLOv11 model loaded (Good balance of accuracy and localization)")
    elif model_choice == "YOLOv12":
        st.info("YOLOv12 model loaded (Attention-based architecture, highest accuracy)")
    elif model_choice == "YOLOv26":
        st.info("YOLOv26 model loaded (NMS-free, optimized for speed)")
    else:
        st.info("Faster R-CNN model loaded (Higher localization accuracy)")
    
    st.markdown("---")
    st.header("Image Upload")
    
    uploaded_files = st.file_uploader(
        "Select images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if st.button("Run Detection", type="primary"):
        if uploaded_files:
            with st.spinner(f"Processing with {model_choice}..."):
                if load_from_files(uploaded_files, model, model_type):
                    st.success(f"Loaded {len(st.session_state.image_paths)} images")
                else:
                    st.error("Failed to load images")
        else:
            st.error("Select at least one image")
    
    st.markdown("---")
    
    if st.session_state.image_paths:
        st.markdown("### Statistics")
        st.markdown(f"**Total images:** {len(st.session_state.image_paths)}")
        st.markdown(f"**Total slugs:** {sum(st.session_state.counts)}")
        st.markdown(f"**Current:** {st.session_state.current_index + 1} / {len(st.session_state.image_paths)}")
        
        st.markdown("---")
        st.markdown(f"**Model:** {model_choice}")
        if len(st.session_state.counts) > 0:
            st.markdown(f"**Average:** {sum(st.session_state.counts)/len(st.session_state.counts):.1f} per image")


# Main area
if st.session_state.image_paths:
    idx = st.session_state.current_index
    
    if 'image_scale' not in st.session_state:
        st.session_state.image_scale = 1.0
    
    # Navigation
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 2, 1, 1, 1, 1])
    with col1:
        if st.button("First", use_container_width=True):
            st.session_state.current_index = 0
            st.rerun()
    with col2:
        if st.button("Prev", use_container_width=True):
            prev_image()
            st.rerun()
    with col3:
        if st.button("Zoom -", use_container_width=True):
            st.session_state.image_scale = max(0.3, st.session_state.image_scale - 0.1)
            st.rerun()
    with col6:
        if st.button("Zoom +", use_container_width=True):
            st.session_state.image_scale = min(2.0, st.session_state.image_scale + 0.1)
            st.rerun()
    with col7:
        if st.button("Next", use_container_width=True):
            next_image()
            st.rerun()
    with col8:
        if st.button("Last", use_container_width=True):
            st.session_state.current_index = len(st.session_state.image_paths) - 1
            st.rerun()
    
    st.caption(f"Zoom: {st.session_state.image_scale:.1f}x")
    st.progress((idx + 1) / len(st.session_state.image_paths))
    
    def resize_image(image_path, scale):
        img = Image.open(image_path)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### Original Image")
        try:
            resized_img = resize_image(st.session_state.image_paths[idx], st.session_state.image_scale)
            st.image(resized_img, use_container_width=False)
        except Exception:
            st.image(st.session_state.image_paths[idx], use_container_width=True)
    
    with col_right:
        st.markdown("### Detection Result")
        try:
            resized_annot = resize_image(st.session_state.annotated_paths[idx], st.session_state.image_scale)
            st.image(resized_annot, use_container_width=False)
        except Exception:
            st.image(st.session_state.annotated_paths[idx], use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"## Slugs detected: **{st.session_state.counts[idx]}**")
    st.markdown("---")
    st.markdown(f"**File:** {st.session_state.filenames[idx]} ({idx + 1} / {len(st.session_state.image_paths)})")
    
else:
    st.info("Select a model, then upload images in the sidebar")