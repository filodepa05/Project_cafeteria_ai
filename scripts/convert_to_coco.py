#!/usr/bin/env python3
"""convert_to_coco.py – Convert datasets to COCO format and export YOLO format.

This script:
1. Converts UNIMIB2016 polygon annotations to COCO bbox format
2. Converts Food-101 to COCO format (using full image as bbox)
3. Merges all datasets into one COCO JSON
4. Exports to YOLO format for Nicolas
5. Filters to only include classes that map to Filo's 43 master categories

Usage:
    # Convert all datasets
    python scripts/convert_to_coco.py --dataset all
    
    # Convert only UNIMIB2016
    python scripts/convert_to_coco.py --dataset unimib2016
    
    # Export to YOLO format
    python scripts/convert_to_coco.py --dataset all --export-yolo
    
    # Split train/val and export YOLO
    python scripts/convert_to_coco.py --dataset all --export-yolo --split 0.8

Output:
    data/processed/
    ├── annotations.json          # Merged COCO format
    ├── images/                   # All images
    └── yolo/                     # YOLO format (if --export-yolo)
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

# Import our utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.class_mapping import (
    FILO_CATEGORIES,
    FILO_TO_IDX,
    UNIMIB_TO_FILO,
    FOOD101_TO_FILO,
    get_filo_class_id,
)
from utils.dataset_utils import (
    polygon_to_bbox,
    coco_bbox_to_yolo,
    load_json,
    save_json,
    copy_image,
    get_image_size,
    merge_coco_datasets,
    filter_coco_by_classes,
    split_train_val,
)


def convert_unimib2016_to_coco(raw_dir: Path, output_dir: Path) -> Path:
    """Convert UNIMIB2016 polygon annotations to COCO format.
    
    UNIMIB2016 structure:
        raw_dir/
        ├── train/                # Training images
        ├── val/                  # Validation images
        ├── test/                 # Test images
        ├── annotationtrain.json  # Training annotations
        ├── annotationval.json    # Validation annotations
        └── annotationtest.json   # Test annotations
    
    Parameters
    ----------
    raw_dir : Path
        Raw dataset directory
    output_dir : Path
        Output directory for processed data
    
    Returns
    -------
    Path
        Path to output COCO JSON file
    """
    print("Converting UNIMIB2016 to COCO format...")
    
    # Check for annotation files at root level
    train_ann = raw_dir / "annotationtrain.json"
    val_ann = raw_dir / "annotationval.json"
    test_ann = raw_dir / "annotationtest.json"
    
    if not train_ann.exists():
        print(f"Warning: Training annotations not found at {train_ann}")
        return None
    
    # COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(FILO_CATEGORIES)],
    }
    
    img_id = 1
    ann_id = 1
    
    # Process train, val, and test splits
    splits = [
        ("train", train_ann, raw_dir / "train"),
        ("val", val_ann, raw_dir / "val"),
        ("test", test_ann, raw_dir / "test"),
    ]
    
    for split_name, ann_file, img_dir in splits:
        if not ann_file.exists():
            print(f"Skipping {split_name} - annotation file not found")
            continue
        
        if not img_dir.exists():
            print(f"Skipping {split_name} - image directory not found")
            continue
        
        print(f"Processing {split_name} split...")
        ann_data = load_json(ann_file)
        
        # Process each image in the annotation file
        for img_key, img_info in tqdm(ann_data.items(), desc=f"Processing {split_name}"):
            # Get image filename
            img_filename = img_info.get("img_name", f"{img_key}.jpg")
            img_path = img_dir / img_filename
            
            if not img_path.exists():
                # Try without extension
                img_path = img_dir / img_key
                if not img_path.exists():
                    continue
            
            try:
                # Get image size
                img_width, img_height = get_image_size(img_path)
                
                # Add image
                coco["images"].append({
                    "id": img_id,
                    "file_name": img_filename,
                    "width": img_width,
                    "height": img_height,
                })
                
                # Process annotations for this image
                for obj in img_info.get("objects", []):
                    label = obj.get("category", "").lower().replace(" ", "_")
                    polygon = obj.get("polygon", [])
                    
                    # Map to Filo class
                    filo_class_id = get_filo_class_id(label, UNIMIB_TO_FILO)
                    if filo_class_id is None:
                        continue  # Skip unmapped classes
                    
                    # Convert polygon to bbox
                    if polygon:
                        bbox = polygon_to_bbox([polygon])
                    else:
                        # Fallback: use image bounds
                        bbox = [0, 0, img_width, img_height]
                    
                    # Estimate portion (UNIMIB doesn't have portions, use default)
                    portion_grams = 150.0
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": filo_class_id,
                        "bbox": bbox,  # [x, y, w, h]
                        "portion_grams": portion_grams,
                    })
                    ann_id += 1
                
                # Copy image to processed directory
                dst_img_path = output_dir / "images" / img_filename
                copy_image(img_path, dst_img_path)
                
                img_id += 1
                
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                continue
    
    # Save COCO JSON
    output_path = output_dir / "unimib2016_coco.json"
    save_json(coco, output_path)
    
    print(f"UNIMIB2016: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return output_path

def convert_food101_to_coco(raw_dir: Path, output_dir: Path, max_images_per_class: int = 100) -> Path:
    """Convert Food-101 to COCO format.
    
    Food-101 structure:
        raw_dir/
        ├── images/               # Images organized in class folders
        ├── meta/
        │   ├── classes.txt      # Class names
        │   ├── train.txt        # Training split
        │   └── test.txt         # Test split
    
    Note: Food-101 has single-label per image (no bounding boxes).
    We create a full-image bbox [0, 0, width, height] for each.
    
    Parameters
    ----------
    raw_dir : Path
        Raw dataset directory
    output_dir : Path
        Output directory for processed data
    max_images_per_class : int
        Maximum images to use per class (to limit dataset size)
    
    Returns
    -------
    Path
        Path to output COCO JSON file
    """
    print("Converting Food-101 to COCO format...")
    
    images_dir = raw_dir / "images"
    
    if not images_dir.exists():
        print(f"Warning: Images directory not found at {images_dir}")
        return None
    
    # COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(FILO_CATEGORIES)],
    }
    
    img_id = 1
    ann_id = 1
    
    # Get all class folders
    class_folders = [d for d in images_dir.iterdir() if d.is_dir()]
    
    for class_folder in tqdm(class_folders, desc="Processing Food-101"):
        class_name = class_folder.name
        
        # Map to Filo class
        filo_class_id = get_filo_class_id(class_name, FOOD101_TO_FILO)
        if filo_class_id is None:
            continue  # Skip unmapped classes
        
        # Get images for this class
        image_files = list(class_folder.glob("*.jpg"))[:max_images_per_class]
        
        for img_path in image_files:
            try:
                # Get image size
                img_width, img_height = get_image_size(img_path)
                
                # Create unique filename
                new_filename = f"food101_{class_name}_{img_path.name}"
                
                # Add image
                coco["images"].append({
                    "id": img_id,
                    "file_name": new_filename,
                    "width": img_width,
                    "height": img_height,
                })
                
                # Add annotation (full image as bbox)
                portion_grams = 150.0  # Default portion
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": filo_class_id,
                    "bbox": [0, 0, img_width, img_height],
                    "portion_grams": portion_grams,
                })
                
                # Copy image
                dst_img_path = output_dir / "images" / new_filename
                copy_image(img_path, dst_img_path)
                
                img_id += 1
                ann_id += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Save COCO JSON
    output_path = output_dir / "food101_coco.json"
    save_json(coco, output_path)
    
    print(f"Food-101: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return output_path


def export_coco_to_yolo(coco: dict, output_dir: Path, train_split: float = 0.8) -> None:
    """Export COCO dataset to YOLO format.
    
    YOLO format:
        output_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
    
    Label format (one .txt per image):
        <class_id> <x_center> <y_center> <width> <height>
    
    Parameters
    ----------
    coco : dict
        COCO dataset
    output_dir : Path
        Output directory
    train_split : float
        Train/validation split ratio
    """
    print("Exporting to YOLO format...")
    
    # Create directories
    yolo_img_train = output_dir / "images" / "train"
    yolo_img_val = output_dir / "images" / "val"
    yolo_lbl_train = output_dir / "labels" / "train"
    yolo_lbl_val = output_dir / "labels" / "val"
    
    for d in [yolo_img_train, yolo_img_val, yolo_lbl_train, yolo_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Split train/val
    train_coco, val_coco = split_train_val(coco, split_ratio=train_split)
    
    # Get image directory
    img_dir = output_dir.parent / "images"
    
    def process_split(coco_split: dict, img_out_dir: Path, lbl_out_dir: Path):
        """Process one split (train or val)."""
        # Group annotations by image
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco_split.get("annotations", []):
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append(ann)
        
        for img in tqdm(coco_split.get("images", []), desc=f"Processing {img_out_dir.name}"):
            img_id = img["id"]
            filename = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]
            
            # Copy image
            src_img = img_dir / filename
            dst_img = img_out_dir / filename
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Create label file
            base_name = Path(filename).stem
            lbl_path = lbl_out_dir / f"{base_name}.txt"
            
            with open(lbl_path, "w") as f:
                for ann in anns_by_image.get(img_id, []):
                    bbox = ann["bbox"]  # [x, y, w, h]
                    class_id = ann["category_id"]
                    
                    # Convert to YOLO format
                    yolo_bbox = coco_bbox_to_yolo(bbox, img_width, img_height)
                    
                    f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    # Process both splits
    process_split(train_coco, yolo_img_train, yolo_lbl_train)
    process_split(val_coco, yolo_img_val, yolo_lbl_val)
    
    # Create data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"# YOLO dataset configuration\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"\n")
        f.write(f"# Number of classes\n")
        f.write(f"nc: {len(FILO_CATEGORIES)}\n")
        f.write(f"\n")
        f.write(f"# Class names\n")
        f.write(f"names: {FILO_CATEGORIES}\n")
    
    print(f"YOLO format exported to {output_dir}")
    print(f"  Train: {len(list(yolo_img_train.glob('*')))} images")
    print(f"  Val: {len(list(yolo_img_val.glob('*')))} images")


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to COCO format")
    parser.add_argument(
        "--dataset",
        choices=["all", "unimib2016", "food101"],
        default="all",
        help="Which dataset to convert",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for data (default: data)",
    )
    parser.add_argument(
        "--export-yolo",
        action="store_true",
        help="Export to YOLO format",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--max-food101-images",
        type=int,
        default=100,
        help="Max images per class for Food-101 (default: 100)",
    )
    
    args = parser.parse_args()
    
    raw_dir = args.data_root / "raw"
    processed_dir = args.data_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    coco_files = []
    
    # Convert UNIMIB2016
    if args.dataset in ("all", "unimib2016"):
        unimib_raw = raw_dir / "unimib2016"
        if unimib_raw.exists():
            coco_path = convert_unimib2016_to_coco(unimib_raw, processed_dir)
            if coco_path:
                coco_files.append(coco_path)
        else:
            print(f"UNIMIB2016 not found at {unimib_raw}")
            print("Run: python scripts/download_datasets.py --dataset unimib2016")
    
    # Convert Food-101
    if args.dataset in ("all", "food101"):
        food101_raw = raw_dir / "food101"
        if food101_raw.exists():
            coco_path = convert_food101_to_coco(
                food101_raw, 
                processed_dir,
                max_images_per_class=args.max_food101_images
            )
            if coco_path:
                coco_files.append(coco_path)
        else:
            print(f"Food-101 not found at {food101_raw}")
            print("Run: python scripts/download_datasets.py --dataset food101")
    
    # Merge datasets
    if len(coco_files) > 1:
        print("\nMerging datasets...")
        merged_path = processed_dir / "annotations.json"
        merged_coco = merge_coco_datasets(coco_files, merged_path)
        print(f"Merged dataset saved to {merged_path}")
        print(f"Total: {len(merged_coco['images'])} images, {len(merged_coco['annotations'])} annotations")
    elif len(coco_files) == 1:
        # Just rename to annotations.json
        merged_path = processed_dir / "annotations.json"
        shutil.copy(coco_files[0], merged_path)
        merged_coco = load_json(merged_path)
    else:
        print("No datasets to merge!")
        return
    
    # Export to YOLO if requested
    if args.export_yolo:
        yolo_dir = processed_dir / "yolo"
        export_coco_to_yolo(merged_coco, yolo_dir, train_split=args.split)
    
    print("\n=== Conversion complete ===")
    print(f"COCO format: {processed_dir / 'annotations.json'}")
    if args.export_yolo:
        print(f"YOLO format: {processed_dir / 'yolo'}")


if __name__ == "__main__":
    main()
