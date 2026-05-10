# detectors/faster_rcnn/train_faster_rcnn_simple.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from collections import defaultdict
import numpy as np

class SlugsDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Загружаем COCO аннотации
        ann_file = self.root_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Файл {ann_file} не найден. Сначала запустите prepare_data_for_faster_rcnn.py")
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Создаем маппинги
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"Loaded {len(self.image_ids)} images for {split}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Загружаем изображение
        img_path = self.root_dir / self.split / "images" / image_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Resize как в YOLO
        original_size = image.size
        target_size = (640, 640)
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Получаем аннотации
        annotations = self.annotations_by_image[image_id]
        boxes = []
        labels = []
        
        # Масштабируем боксы
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([
                x * scale_x,
                y * scale_y,
                (x + w) * scale_x,
                (y + h) * scale_y
            ])
            labels.append(ann['category_id'])
        
        # Конвертируем в тензоры
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([image_id]),
            'area': (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0]) if len(boxes_tensor) > 0 else torch.tensor([0.0]),
            'iscrowd': torch.zeros((len(boxes_tensor),), dtype=torch.int64),
        }
        
        image_tensor = F.to_tensor(image)
        return image_tensor, target

def train_faster_rcnn():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Пути
    current_dir = Path(__file__).parent
    coco_root = current_dir / "coco_dataset"
    
    if not coco_root.exists():
        print(f"\n❌ Ошибка: {coco_root} не существует!")
        print("Запустите python detectors/faster_rcnn/prepare_data_for_faster_rcnn.py")
        return None
    
    # Загружаем датасеты
    try:
        train_dataset = SlugsDataset(coco_root, "train")
        val_dataset = SlugsDataset(coco_root, "val")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return None
    
    if len(train_dataset) == 0:
        print("\n❌ Нет данных для обучения!")
        return None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Меньше batch для CPU
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Для Windows лучше 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=4
    )
    
    # Создаем модель
    backbone = torchvision.models.mobilenet_v2(weights='DEFAULT').features
    backbone.out_channels = 1280
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=2,  # background + slug
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=640,
        max_size=640
    )
    
    model.to(device)
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    
    print("\n" + "="*60)
    print("Обучение Faster R-CNN")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(150):  # 30 эпох для теста
        model.train()
        train_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_losses.append(losses.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx:3d} | Loss: {losses.item():.4f}")
        
        avg_loss = np.mean(train_losses)
        print(f"\nEpoch {epoch+1}/30 | Avg Loss: {avg_loss:.4f}")
        
        # Сохраняем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = current_dir / "faster_rcnn_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")
        
        print("-" * 40)
    
    print(f"\n✅ Обучение завершено! Модель сохранена в {current_dir}/faster_rcnn_best.pth")
    return model

if __name__ == "__main__":
    train_faster_rcnn()