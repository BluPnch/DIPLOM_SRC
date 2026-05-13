# detectors/faster_rcnn/train_faster_rcnn_improved.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from collections import defaultdict
import numpy as np
import albumentations as A


def collate_fn(batch):
    return tuple(zip(*batch))


def clip_bboxes(bboxes, image_shape):
    """Обрезает координаты боксов, чтобы они не выходили за границы изображения"""
    height, width = image_shape[:2]
    clipped = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(x_min, width))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height))
        y_max = max(0, min(y_max, height))
        if x_min < x_max and y_min < y_max:
            clipped.append([x_min, y_min, x_max, y_max])
    return clipped


class SlugsDatasetAugmented(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        
        ann_file = self.root_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Файл {ann_file} не найден")
        
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        self.image_ids = list(self.images.keys())
        print(f"Loaded {len(self.image_ids)} images for {split}")
        
        # Аугментации с safe_bbox_params
        if split == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3, border_mode=0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            ], bbox_params=A.BboxParams(
                format='coco', 
                label_fields=['labels'],
                min_visibility=0.3,
                min_area=25
            ))
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        img_path = self.root_dir / self.split / "images" / image_info['file_name']
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        target_size = (640, 640)
        
        annotations = self.annotations_by_image[image_id]
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Применяем аугментации
        if self.transform is not None and len(boxes) > 0:
            image_np = np.array(image)
            try:
                transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
                image = Image.fromarray(transformed['image'])
                boxes = transformed['bboxes']
                labels = transformed['labels']
                
                # Обрезаем боксы, которые вышли за границы
                boxes = clip_bboxes(boxes, image.size)
            except Exception as e:
                pass
        
        # Resize
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        
        # Масштабируем боксы
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        
        boxes_tensor = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_tensor.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
        
        boxes_tensor = torch.as_tensor(boxes_tensor, dtype=torch.float32) if boxes_tensor else torch.zeros((0, 4), dtype=torch.float32)
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
    # CUDA проверка
    if not torch.cuda.is_available():
        print("❌ CUDA не доступна! Обучение будет на CPU (очень медленно)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    current_dir = Path(__file__).parent
    coco_root = current_dir / "coco_dataset"
    
    if not coco_root.exists():
        print(f"Ошибка: {coco_root} не существует!")
        return None
    
    train_dataset = SlugsDatasetAugmented(coco_root, "train")
    val_dataset = SlugsDatasetAugmented(coco_root, "val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )
    
    
    backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT').features
    backbone.out_channels = 576
    
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
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
    
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    print("\n" + "="*60)
    print("Обучение Faster R-CNN (с аугментациями)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(150):
        model.train()
        train_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(losses.item())
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx:3d} | Loss: {losses.item():.4f}")
        
        avg_loss = np.mean(train_losses)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/150 | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = current_dir / "faster_rcnn_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print("-" * 40)
    
    print(f"\n✅ Обучение завершено! Модель сохранена в {current_dir}/faster_rcnn_best.pth")
    print(f"   Финальная loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train_faster_rcnn()