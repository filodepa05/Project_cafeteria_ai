#!/usr/bin/env python3
"""download_datasets.py – Download UNIMIB2016 and Food-101 datasets.

Usage:
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --dataset unimib2016
    python scripts/download_datasets.py --dataset food101

Note: Requires kaggle CLI to be installed and configured:
    pip install kaggle
    # Place kaggle.json in ~/.kaggle/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> None:
    """Download a dataset from Kaggle.
    
    Parameters
    ----------
    dataset_name : str
        Kaggle dataset name (e.g., "dangvanthuc0209/unimib2016")
    output_dir : Path
        Directory to download to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {dataset_name}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)],
            check=True,
        )
        
        # Find and extract zip file
        zip_files = list(output_dir.glob("*.zip"))
        if zip_files:
            zip_path = zip_files[0]
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            zip_path.unlink()  # Remove zip file
            print(f"Downloaded and extracted to {output_dir}")
        else:
            print(f"No zip file found in {output_dir}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Make sure kaggle CLI is installed and authenticated:")
        print("  pip install kaggle")
        print("  # Place kaggle.json in ~/.kaggle/")
        sys.exit(1)


def download_unimib2016(data_root: Path) -> Path:
    """Download UNIMIB2016 dataset.
    
    Source: 1,027 real canteen tray photos, 73 food classes, polygon annotations
    """
    output_dir = data_root / "raw" / "unimib2016"
    download_kaggle_dataset("dangvanthuc0209/unimib2016", output_dir)
    return output_dir


def download_food101(data_root: Path) -> Path:
    """Download Food-101 dataset.
    
    Source: 101,000 images, 101 classes
    """
    output_dir = data_root / "raw" / "food101"
    download_kaggle_dataset("dansbecker/food-101", output_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Smart Tray")
    parser.add_argument(
        "--dataset",
        choices=["all", "unimib2016", "food101"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for data (default: data)",
    )
    
    args = parser.parse_args()
    
    print(f"Data root: {args.data_root.absolute()}")
    
    if args.dataset in ("all", "unimib2016"):
        print("\n=== Downloading UNIMIB2016 ===")
        download_unimib2016(args.data_root)
    
    if args.dataset in ("all", "food101"):
        print("\n=== Downloading Food-101 ===")
        download_food101(args.data_root)
    
    print("\n=== Download complete ===")
    print(f"Datasets saved to {args.data_root / 'raw'}")
    print("\nNext step: Run scripts/convert_to_coco.py to convert to COCO format")


if __name__ == "__main__":
    main()
