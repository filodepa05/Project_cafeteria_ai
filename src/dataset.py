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
  "categories": [{"id": 0, "name": "pasta"}, ...]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from src.config import Config


# ── FOOD CATEGORIES (order matters — index = class id) ────────────
#
# Derived from:
#   • IE Tower weekly rotating menu (Oct 2024)
#   • Do Eat! cafeteria menu (María de Molina)
#   • Campus Segovia Eurest menu (Autumn 2023)
#   • UNIMIB2016 class overlap (for training data coverage)
#
# Naming conventions:
#   - Generic preparation names, not recipe names
#     ("grilled_chicken" not "pollo_al_ajillo")
#   - Snake_case, ASCII only
#   - 43 classes total
#
CATEGORIES: list[str] = [
    # ── Starches & grains (0–7) ───────────────────────────────────
    "pasta",            # 0  tallarines, farfalle, fusilli, penne, macaroni, spaghetti, ravioli
    "rice",             # 1  basmati, arroz con pollo, risotto, paella, arroz tres delicias
    "pizza",            # 2  margherita, quattro formaggi, BBQ, arugula, pan pizza
    "bread",            # 3  baguette, pan, crusty bread
    "fries",            # 4  patatas fritas thin-cut (fast food style)
    "couscous",         # 5  taboulé, cous-cous de verduras, cous-cous de garbanzos
    "potatoes",         # 6  boiled / mashed / stewed potatoes, patatas con carne
    "wrap_sandwich",    # 7  wrap, burrito, sandwich, bocadillo, tortilla wrap

    # ── Poultry (8–11) ────────────────────────────────────────────
    "grilled_chicken",  # 8  pechuga plancha, contramuslos grill, pollo al ajillo
    "fried_chicken",    # 9  pollo empanado, alitas fritas, chicken burger
    "chicken_stew",     # 10 pollo guisado, pollo a la moruna, arroz con pollo
    "turkey",           # 11 chuleta de pavo, muslos de pavo, pavo a la catalana

    # ── Beef & pork (12–16) ───────────────────────────────────────
    "grilled_beef",     # 12 entrecot, entraña grill, filete de ternera plancha
    "beef_stew",        # 13 ragout de ternera, goulash, ternera con verduras, chili con carne
    "meatballs",        # 14 albóndigas al curry, albóndigas con setas
    "grilled_pork",     # 15 secreto ibérico, solomillo de cerdo, lomo plancha, chuletas
    "pork_ribs",        # 16 costilla caramelizada, costillas horneadas

    # ── Fish (17–22) ──────────────────────────────────────────────
    "salmon",           # 17 salmón grill, salmón al eneldo, salmón papillote
    "hake",             # 18 merluza plancha, merluza cajún, merluza crujiente
    "tuna",             # 19 emperador plancha, atún miel-mostaza, atún a la plancha
    "cod",              # 20 bacalao rebozado, bacalao ali-oli, bacalao gratinado
    "grilled_fish",     # 21 generic grilled white fish: perca, lubina, rodaballo, rape
    "fried_fish",       # 22 pescaditos fritos, adobito empanado, calamares andaluza

    # ── Eggs (23) ─────────────────────────────────────────────────
    "eggs",             # 23 huevos fritos, tortilla francesa, revuelto, huevos rotos

    # ── Legumes & pulses (24–25) ──────────────────────────────────
    "lentils",          # 24 lentejas estofadas, salteado arroz y lentejas, mujadara
    "chickpeas",        # 25 chana masala, garbanzos al curry, cous-cous de garbanzos

    # ── Vegetables (26–31) ────────────────────────────────────────
    "salad",            # 26 ensalada mixta, caesar, ensalada de la casa, ensalada vitaminica
    "soup_cream",       # 27 crema de champiñones, crema de calabaza, velouté, sopa de tomate
    "grilled_vegetables", # 28 pimientos asados, calabacines asados, tomates asados, ratatouille
    "sauteed_vegetables", # 29 judías verdes salteadas, guisantes salteados, setas al ajillo
    "broccoli",         # 30 brócoli gratinado, brócoli chowder, crema de brócoli
    "stuffed_peppers",  # 31 pimientos rellenos de atún, pimientos rellenos vegetarianos

    # ── Poke & bowls (32) ─────────────────────────────────────────
    "poke_bowl",        # 32 poke bowl de atún, poke bowl de salmón (Do Eat)

    # ── Lasagne & baked pasta (33) ────────────────────────────────
    "lasagne",          # 33 lasaña de carne, lasaña vegetal, canelones, musaka

    # ── Fruit (34–35) ─────────────────────────────────────────────
    "fresh_fruit",      # 34 fruta fresca, vaso de fruta variada (dessert option every day)
    "fruit_salad",      # 35 macedonia de frutas, ensalada de frutas

    # ── Dairy & desserts (36–38) ──────────────────────────────────
    "yogurt",           # 36 yogurt natural, natillas, lácteos (daily dessert option)
    "cake_pastry",      # 37 tarta, brownie, milhojas, profiteroles, croissant, muffin, cookie
    "ice_cream_sorbet", # 38 sorbete, batido de fruta, helado

    # ── Drinks (39) ───────────────────────────────────────────────
    "juice_drink",      # 39 zumo, smoothie, zumo vitamínico (Do Eat / IE Tower)

    # ── Always-present IE staples (40–42) ─────────────────────────
    "rotisserie_chicken",  # 40 pollo asado entero al horno / pollo al ast — whole roasted bird,
                           #    visually distinct from grilled breast or stew (golden skin, full carcass)
    "fried_potatoes",      # 41 patatas bravas, patatas rústicas fritas, thick-cut wedges —
                           #    chunkier & crispier than fries (class 4)
    "baked_potatoes",      # 42 patatas rústicas al horno, patatas gratinadas, patatas panaderas —
                           #    oven-roasted, drier surface, no deep-fry colour
]

# Reverse lookup: name → class id
CAT_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

# Total number of classes — use this instead of hardcoding 40
NUM_CLASSES: int = len(CATEGORIES)   # 43


# ═══════════════════════════════════════════════════════════════════
#  Albumentations transforms
# ═══════════════════════════════════════════════════════════════════

def get_train_transforms(image_size: int = 640) -> A.Compose:
    """Training transforms with augmentation using albumentations.
    
    Includes:
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation)
    - Random crop
    - Rotation
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.RandomCrop(height=int(image_size * 0.9), width=int(image_size * 0.9), p=0.3),
        A.Resize(image_size, image_size),  # Resize back after crop
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'portions']))


def get_val_transforms(image_size: int = 640) -> A.Compose:
    """Validation transforms without augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'portions'], clip=True))


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
        self.transform = transform

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
        img_array = np.array(img)

        orig_h, orig_w = img_array.shape[:2]

        # Get annotations
        anns = self._anns_by_image.get(img_info["id"], [])
        boxes, labels, portions = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # xyxy format for albumentations
            labels.append(ann["category_id"])
            portions.append(ann.get("portion_grams", 100.0))

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(
                image=img_array,
                bboxes=boxes,
                labels=labels,
                portions=portions,
            )
            img_tensor = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
            portions = transformed["portions"]
        else:
            # Default: just resize and normalize
            transform = get_val_transforms(self.image_size)
            transformed = transform(
                image=img_array,
                bboxes=boxes,
                labels=labels,
                portions=portions,
            )
            img_tensor = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
            portions = transformed["portions"]

        # Handle empty boxes after augmentation
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
            portions = torch.zeros(0, dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            portions = torch.tensor(portions, dtype=torch.float32)

        return {
            "image": img_tensor,
            "boxes": boxes,
            "labels": labels,
            "portions": portions,
        }


# ═══════════════════════════════════════════════════════════════════
#  Synthetic dataset (for smoke tests / debug)
# ═══════════════════════════════════════════════════════════════════

class SyntheticTrayDataset(Dataset):
    """Generates random images + fake annotations.  No files needed."""

    def __init__(self, n_samples: int = 32, image_size: int = 224, num_classes: int = NUM_CLASSES):
        self.n = n_samples
        self.size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.RandomState(idx)
        torch.manual_seed(idx)
        img = torch.randn(3, self.size, self.size)

        n_objs = rng.randint(1, 4)
        boxes = []
        for _ in range(n_objs):
            x1 = rng.randint(0, self.size // 2)
            y1 = rng.randint(0, self.size // 2)
            x2 = rng.randint(x1 + 20, min(x1 + self.size // 2, self.size))
            y2 = rng.randint(y1 + 20, min(y1 + self.size // 2, self.size))
            boxes.append([x1, y1, x2, y2])

        labels = rng.randint(0, self.num_classes, size=n_objs)
        portions = rng.uniform(50, 300, size=n_objs)

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
            transform=None,  # Will be set after split
        )

    n_train = int(len(full_ds) * cfg.data.train_split)
    n_val = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # Apply different transforms to train and val
    # Access the underlying dataset through .dataset attribute
    train_ds.dataset.transform = get_train_transforms(cfg.data.image_size)
    val_ds.dataset.transform = get_val_transforms(cfg.data.image_size)

    shared = dict(
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, **shared)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, **shared)

    return train_loader, val_loader