"""
dataset.py – TrayDataset for food detection + portion estimation.

Supports two modes:
  • Real data  : COCO-format annotations.json + image folder
  • Synthetic  : random tensors + fake labels (for smoke tests)

Annotation schema (extends COCO):
{
  "images": [{"id": 1, "file_name": "tray_0001.jpg", "width": 640, "height": 640}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 3,
      "bbox": [x, y, w, h],        # COCO format
      "portion_grams": 185.0        # ← custom field
    }
  ],
  "categories": [{"id": 0, "name": "banana"}, ...]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from PIL import Image

from src.config import Config


# ── FOOD CATEGORIES (order matters — index = class id) ────────────
CATEGORIES: list[str] = [
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot_dog", "pizza", "donut", "cake",
]

CAT_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}


# ═══════════════════════════════════════════════════════════════════
#  Real dataset
# ═══════════════════════════════════════════════════════════════════

class TrayDataset(Dataset):
    """Loads tray images with bounding-box + portion-gram annotations."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 640,
        transform: Any | None = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.image_size = image_size
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        ann_path = self.root / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotations not found at {ann_path}.  "
                f"Use SyntheticTrayDataset for debug runs."
            )

        with open(ann_path) as f:
            coco = json.load(f)

        # Index images and annotations by image_id
        self.images: list[dict] = coco["images"]
        self._anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_info = self.images[idx]
        img_path = self.image_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        img_tensor = self.transform(img)

        # Scale bboxes to resized coordinates
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h

        anns = self._anns_by_image.get(img_info["id"], [])
        boxes, labels, portions = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy])  # xyxy
            labels.append(ann["category_id"])
            portions.append(ann.get("portion_grams", 100.0))

        return {
            "image": img_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "portions": torch.tensor(portions, dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════
#  Synthetic dataset (for smoke tests / debug)
# ═══════════════════════════════════════════════════════════════════

class SyntheticTrayDataset(Dataset):
    """Generates random images + fake annotations.  No files needed."""

    def __init__(self, n_samples: int = 32, image_size: int = 224, num_classes: int = 10):
        self.n = n_samples
        self.size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = torch.randn(3, self.size, self.size)

        n_objs = np.random.randint(1, 4)
        boxes = []
        for _ in range(n_objs):
            x1 = np.random.randint(0, self.size // 2)
            y1 = np.random.randint(0, self.size // 2)
            x2 = np.random.randint(x1 + 20, min(x1 + self.size // 2, self.size))
            y2 = np.random.randint(y1 + 20, min(y1 + self.size // 2, self.size))
            boxes.append([x1, y1, x2, y2])

        labels = np.random.randint(0, self.num_classes, size=n_objs)
        portions = np.random.uniform(50, 300, size=n_objs)

        return {
            "image": img,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "portions": torch.tensor(portions, dtype=torch.float32),
        }


# ═══════════════════════════════════════════════════════════════════
#  Collate + DataLoader factory
# ═══════════════════════════════════════════════════════════════════

def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate: images stack, boxes/labels/portions stay as lists
    (variable number of objects per image)."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "portions": [b["portions"] for b in batch],
    }


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) based on config."""
    if cfg._debug.use_synthetic:
        full_ds = SyntheticTrayDataset(
            n_samples=cfg._debug.synthetic_samples,
            image_size=cfg.data.image_size,
            num_classes=cfg.model.num_classes,
        )
    else:
        full_ds = TrayDataset(
            root=cfg.data.root,
            image_size=cfg.data.image_size,
        )

    n_train = int(len(full_ds) * cfg.data.train_split)
    n_val = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    shared = dict(
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, **shared)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, **shared)

    return train_loader, val_loader