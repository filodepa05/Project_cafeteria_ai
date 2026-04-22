"""dataset_utils.py – Shared utilities for dataset conversion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def polygon_to_bbox(polygon: list[list[float]]) -> list[float]:
    """Convert polygon [[x1,y1, x2,y2, ...]] to COCO bbox [x, y, w, h].
    
    Parameters
    ----------
    polygon : list of lists
        Polygon coordinates as [[x1, y1, x2, y2, ...]]
    
    Returns
    -------
    list[float]
        COCO format bbox [x, y, width, height]
    """
    # Flatten polygon coordinates
    coords = polygon[0] if isinstance(polygon[0], list) else polygon
    xs = coords[::2]  # Even indices are x
    ys = coords[1::2]  # Odd indices are y
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def bbox_xyxy_to_xywh(bbox: list[float]) -> list[float]:
    """Convert bbox from xyxy to xywh format.
    
    Parameters
    ----------
    bbox : list[float]
        [x1, y1, x2, y2] format
    
    Returns
    -------
    list[float]
        [x, y, width, height] format
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert bbox from xywh to xyxy format.
    
    Parameters
    ----------
    bbox : list[float]
        [x, y, width, height] format
    
    Returns
    -------
    list[float]
        [x1, y1, x2, y2] format
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def coco_bbox_to_yolo(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """Convert COCO bbox to YOLO format.
    
    Parameters
    ----------
    bbox : list[float]
        COCO format [x, y, width, height] in pixels
    img_width : int
        Image width
    img_height : int
        Image height
    
    Returns
    -------
    list[float]
        YOLO format [x_center, y_center, width, height] normalized 0-1
    """
    x, y, w, h = bbox
    
    # Normalize
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]


def copy_image(src_path: Path, dst_path: Path) -> None:
    """Copy image file, creating destination directory if needed."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_image_size(image_path: Path) -> tuple[int, int]:
    """Get image dimensions (width, height)."""
    with Image.open(image_path) as img:
        return img.size


def merge_coco_datasets(coco_files: list[Path], output_path: Path) -> dict:
    """Merge multiple COCO datasets into one.
    
    Parameters
    ----------
    coco_files : list[Path]
        Paths to COCO JSON files
    output_path : Path
        Output path for merged dataset
    
    Returns
    -------
    dict
        Merged COCO dataset
    """
    merged = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    # Track ID offsets
    img_id_offset = 0
    ann_id_offset = 0
    
    for coco_path in coco_files:
        coco = load_json(coco_path)
        
        # Get ID mappings for this dataset
        img_id_map = {}
        for img in coco.get("images", []):
            old_id = img["id"]
            new_id = old_id + img_id_offset
            img_id_map[old_id] = new_id
            img["id"] = new_id
            merged["images"].append(img)
        
        for ann in coco.get("annotations", []):
            old_id = ann["id"]
            new_id = old_id + ann_id_offset
            ann["id"] = new_id
            ann["image_id"] = img_id_map[ann["image_id"]]
            merged["annotations"].append(ann)
        
        # Update offsets
        if coco.get("images"):
            img_id_offset = max(img["id"] for img in merged["images"]) + 1
        if coco.get("annotations"):
            ann_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1
        
        # Use categories from first dataset (they should all match)
        if not merged["categories"] and coco.get("categories"):
            merged["categories"] = coco["categories"]
    
    save_json(merged, output_path)
    return merged


def filter_coco_by_classes(coco: dict, valid_class_ids: set[int]) -> dict:
    """Filter COCO dataset to keep only specified classes.
    
    Parameters
    ----------
    coco : dict
        COCO dataset
    valid_class_ids : set[int]
        Set of valid category IDs to keep
    
    Returns
    -------
    dict
        Filtered COCO dataset
    """
    filtered = {
        "images": [],
        "annotations": [],
        "categories": [cat for cat in coco.get("categories", []) if cat["id"] in valid_class_ids],
    }
    
    # Keep only annotations for valid classes
    valid_anns = [ann for ann in coco.get("annotations", []) if ann["category_id"] in valid_class_ids]
    
    # Get images that have at least one valid annotation
    valid_img_ids = {ann["image_id"] for ann in valid_anns}
    filtered["images"] = [img for img in coco.get("images", []) if img["id"] in valid_img_ids]
    filtered["annotations"] = valid_anns
    
    return filtered


def split_train_val(coco: dict, split_ratio: float = 0.8, seed: int = 42) -> tuple[dict, dict]:
    """Split COCO dataset into train and validation sets.
    
    Parameters
    ----------
    coco : dict
        COCO dataset
    split_ratio : float
        Ratio for train set (default 0.8)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple[dict, dict]
        (train_coco, val_coco)
    """
    np.random.seed(seed)
    
    images = coco.get("images", [])
    n_train = int(len(images) * split_ratio)
    
    indices = np.random.permutation(len(images))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_img_ids = {images[i]["id"] for i in train_indices}
    val_img_ids = {images[i]["id"] for i in val_indices}
    
    def make_split(img_ids: set[int]) -> dict:
        return {
            "images": [img for img in images if img["id"] in img_ids],
            "annotations": [ann for ann in coco.get("annotations", []) if ann["image_id"] in img_ids],
            "categories": coco.get("categories", []),
        }
    
    return make_split(train_img_ids), make_split(val_img_ids)
