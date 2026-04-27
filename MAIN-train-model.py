import os
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Ограничение потоков PyTorch
torch.set_num_threads(1)

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=2,
    workers=0,
    project="runs/train",
    name="1_experiment",
    amp=False
)