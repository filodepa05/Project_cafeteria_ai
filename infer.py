#!/usr/bin/env python3
"""
infer.py – Inference entry point.

Usage:
    python infer.py path/to/tray.jpg --checkpoint checkpoints/best.pt
    python infer.py path/to/tray.jpg --checkpoint checkpoints/best.pt --output result.json
    python infer.py path/to/tray.jpg --auto-checkpoint   # finds best in checkpoints/
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.inference import TrayInferencePipeline
from src.utils.io import find_best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Smart Tray inference on a tray image.")
    parser.add_argument("image", type=str, help="Path to the tray image")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .pt checkpoint")
    parser.add_argument("--auto-checkpoint", action="store_true",
                        help="Automatically pick the best checkpoint from save_dir")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config (must match training config)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save JSON to this file instead of printing to stdout")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override confidence threshold (0-1)")
    args = parser.parse_args()

    # ── Validate image ────────────────────────────────────────────
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found – {img_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────
    overrides = {}
    if args.threshold is not None:
        overrides["confidence_threshold"] = args.threshold
    cfg = load_config(args.config, overrides if overrides else None)

    # ── Resolve checkpoint ────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.auto_checkpoint:
        ckpt_path = find_best_checkpoint(cfg.checkpoint.save_dir)
        print(f"  Auto-selected checkpoint: {ckpt_path}", file=sys.stderr)
    else:
        print("Error: provide --checkpoint PATH or --auto-checkpoint", file=sys.stderr)
        sys.exit(1)

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found – {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────
    print(f"  Loading model from {ckpt_path.name} …", file=sys.stderr)
    pipeline = TrayInferencePipeline(cfg, ckpt_path)

    print(f"  Analysing {img_path.name} …", file=sys.stderr)
    result_json = pipeline.run_to_json(img_path)

    if args.output:
        Path(args.output).write_text(result_json)
        print(f"  Saved to {args.output}", file=sys.stderr)
    else:
        print(result_json)


if __name__ == "__main__":
    main()