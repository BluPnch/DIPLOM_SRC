import os
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset_yaml  # noqa: E402

st.set_page_config(page_title="Детектор слизней", layout="wide")
st.title("🐌 Детектор слизней на изображениях")


@st.cache_resource
def load_model():
    weights = dataset_yaml.resolve_yolo_best_weights(PROJECT_ROOT)
    if weights is not None and weights.is_file():
        st.success(f"✅ Загружена модель: {weights}")
        return YOLO(str(weights))
    st.warning(
        "⚠️ Не найден best.pt (искали runs/detect/..., detectors/yolo/runs/..., прочие runs/**/weights/)"
    )
    st.info(
        "🔄 Использую предобученную модель yolo11n.pt (может находить не только слизней)"
    )
    return YOLO("yolo11n.pt")


model = load_model()

if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "annotated_paths" not in st.session_state:
    st.session_state.annotated_paths = []
if "counts" not in st.session_state:
    st.session_state.counts = []
if "filenames" not in st.session_state:
    st.session_state.filenames = []


def process_images(image_paths):
    st.session_state.image_paths = []
    st.session_state.annotated_paths = []
    st.session_state.counts = []
    st.session_state.filenames = []
    st.session_state.current_index = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, img_path in enumerate(image_paths):
        status_text.text(f"Обработка {i+1}/{len(image_paths)}: {Path(img_path).name}")

        st.session_state.image_paths.append(img_path)
        st.session_state.filenames.append(Path(img_path).name)

        results = model.predict(source=img_path, conf=0.25, device="cpu", verbose=False)

        for r in results:
            annotated = r.plot()
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(temp_path, annotated)
            st.session_state.annotated_paths.append(temp_path)

            num = len(r.boxes) if r.boxes is not None else 0
            st.session_state.counts.append(num)

        progress_bar.progress((i + 1) / len(image_paths))

    status_text.text(f"✅ Обработано {len(image_paths)} изображений")
    return len(image_paths) > 0


def load_from_folder(folder_path):
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = []

    for ext in extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if not image_files:
        st.error(f"❌ В папке {folder_path} нет изображений")
        return False

    return process_images([str(f) for f in image_files])


def load_from_files(files):
    if not files:
        return False

    temp_dir = tempfile.mkdtemp()
    image_paths = []

    for file in files:
        img_path = os.path.join(temp_dir, file.name)
        with open(img_path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(img_path)

    return process_images(image_paths)


def next_image():
    if st.session_state.current_index < len(st.session_state.image_paths) - 1:
        st.session_state.current_index += 1


def prev_image():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1


with st.sidebar:
    st.header("📂 Загрузка изображений")

    folder_path = st.text_input("Или укажите путь к папке:", placeholder="C:/путь/к/папке")

    if st.button("📁 Загрузить папку", type="primary"):
        if folder_path and os.path.exists(folder_path):
            with st.spinner("Загрузка и обработка..."):
                if load_from_folder(folder_path):
                    st.success(f"✅ Загружено {len(st.session_state.image_paths)} изображений")
                else:
                    st.error("❌ Не удалось загрузить изображения")
        else:
            st.error("❌ Укажите корректный путь к папке")

    st.markdown("---")
    st.markdown("### Или выберите файлы")

    uploaded_files = st.file_uploader(
        "Выберите изображения",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("🖼️ Загрузить файлы"):
        if uploaded_files:
            with st.spinner("Загрузка и обработка..."):
                if load_from_files(uploaded_files):
                    st.success(f"✅ Загружено {len(st.session_state.image_paths)} изображений")
                else:
                    st.error("❌ Не удалось загрузить изображения")
        else:
            st.error("❌ Выберите хотя бы один файл")

    st.markdown("---")

    if st.session_state.image_paths:
        st.markdown("### 📊 Статистика")
        st.markdown(f"**Всего изображений:** {len(st.session_state.image_paths)}")
        st.markdown(f"**Всего слизней:** {sum(st.session_state.counts)}")
        st.markdown(f"**Текущее:** {st.session_state.current_index + 1} / {len(st.session_state.image_paths)}")


if st.session_state.image_paths:
    idx = st.session_state.current_index

    if "image_scale" not in st.session_state:
        st.session_state.image_scale = 1.0

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 2, 1, 1, 1, 1])
    with col1:
        if st.button("⏮️ В начало", use_container_width=True):
            st.session_state.current_index = 0
            st.rerun()
    with col2:
        if st.button("◀ Назад", use_container_width=True):
            prev_image()
            st.rerun()
    with col3:
        if st.button("🔍 -", use_container_width=True):
            st.session_state.image_scale = max(0.3, st.session_state.image_scale - 0.1)
            st.rerun()
    with col6:
        if st.button("🔍 +", use_container_width=True):
            st.session_state.image_scale = min(2.0, st.session_state.image_scale + 0.1)
            st.rerun()
    with col7:
        if st.button("Вперёд ▶", use_container_width=True):
            next_image()
            st.rerun()
    with col8:
        if st.button("⏭️ В конец", use_container_width=True):
            st.session_state.current_index = len(st.session_state.image_paths) - 1
            st.rerun()

    st.caption(f"🔍 Масштаб: {st.session_state.image_scale:.1f}x")

    st.progress((idx + 1) / len(st.session_state.image_paths))

    def resize_image(image_path, scale):
        img = Image.open(image_path)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📷 Исходное изображение")
        try:
            resized_img = resize_image(st.session_state.image_paths[idx], st.session_state.image_scale)
            st.image(resized_img, use_container_width=False)
        except Exception:
            st.image(st.session_state.image_paths[idx], use_container_width=True)

    with col_right:
        st.markdown("### 🐍 Результат детекции")
        try:
            resized_annot = resize_image(st.session_state.annotated_paths[idx], st.session_state.image_scale)
            st.image(resized_annot, use_container_width=False)
        except Exception:
            st.image(st.session_state.annotated_paths[idx], use_container_width=True)

    st.markdown("---")
    st.markdown(f"## 🐌 Количество слизней: **{st.session_state.counts[idx]}**")
    st.markdown("---")
    st.markdown(f"**📄 {st.session_state.filenames[idx]}** ({idx + 1} / {len(st.session_state.image_paths)})")

else:
    st.info("👈 Выберите папку или загрузите файлы в боковой панели")
