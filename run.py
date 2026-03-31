#!/usr/bin/env python3
"""
run.py – CLI entry point for the food-tray estimator.

Usage:
    python run.py path/to/tray_photo.jpg
    python run.py path/to/tray_photo.jpg --output result.json
"""

import argparse
import sys
from pathlib import Path

from pipeline import TrayPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Estimate calories and macros from a food-tray image."
    )
    parser.add_argument("image", type=str, help="Path to the tray image")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional: save JSON to this file instead of printing",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found – {img_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model and analyzing {img_path.name} …", file=sys.stderr)
    pipe = TrayPipeline()
    result_json = pipe.analyze_to_json(img_path)

    if args.output:
        Path(args.output).write_text(result_json)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(result_json)


if __name__ == "__main__":
    main()