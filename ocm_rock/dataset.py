import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class LabelMeMaskDataset(Dataset):
    """读取 LabelMe polygon 标注，构造 Mask R-CNN 所需 targets。"""
    def __init__(self, root: str, split: str = "train", transforms=None):
        self.root = Path(root)
        self.img_dir = self.root / split / "images"
        self.ann_dir = self.root / split / "annotations"
        self.transforms = transforms
        self.images = sorted([p for p in self.img_dir.glob("*.png")])
        if not self.images:
            self.images = sorted([p for p in self.img_dir.glob("*.jpg")])
        if not self.images:
            raise FileNotFoundError(f"没有找到图像: {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        ann_path = self.ann_dir / (img_path.stem + ".json")
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        masks = []
        if ann_path.exists():
            data = json.load(open(ann_path, "r", encoding="utf-8"))
            for shape in data.get("shapes", []):
                if shape.get("shape_type", "polygon") != "polygon":
                    continue
                pts = [tuple(p) for p in shape["points"]]
                if len(pts) < 3:
                    continue
                m = Image.new("L", (W, H), 0)
                ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
                mask = np.array(m, dtype=np.uint8)
                if mask.sum() > 10:
                    masks.append(mask)
        if len(masks) == 0:
            masks = np.zeros((0, H, W), dtype=np.uint8)
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            masks = np.stack(masks, axis=0)
            boxes = []
            for m in masks:
                ys, xs = np.where(m > 0)
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            boxes = np.array(boxes, dtype=np.float32)

        image = torch.as_tensor(np.array(img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else [], dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
