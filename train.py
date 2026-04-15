#!/usr/bin/env python3
"""
train.py – Training entry point.

Usage:
    # Debug run (synthetic data, 2 epochs, CPU)
    python train.py --config configs/experiment/debug.yaml

    # Full training
    python train.py --config configs/base.yaml

    # Override any param from CLI
    python train.py --config configs/base.yaml --epochs 50 --lr 0.0003
"""

import argparse
import sys

from src.config import load_config
from src.dataset import build_dataloaders
from src.models.tray_model import TrayModel
from src.trainer import Trainer


def parse_args() -> tuple[str, dict]:
    """Parse --config path and any flat overrides (--key value)."""
    parser = argparse.ArgumentParser(description="Train the Smart Tray model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config file")

    # Allow arbitrary --key value overrides
    known, unknown = parser.parse_known_args()

    overrides: dict = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val_str = unknown[i + 1]
                # Auto-cast to int/float/bool
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = {"true": True, "false": False}.get(val_str.lower(), val_str)
                overrides[key] = val
                i += 2
                continue
        i += 1

    return known.config, overrides


def main() -> None:
    config_path, overrides = parse_args()

    print(f"  Loading config: {config_path}")
    if overrides:
        print(f"  CLI overrides:  {overrides}")

    cfg = load_config(config_path, overrides)

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Dataset:  {len(train_loader.dataset)} train  /  {len(val_loader.dataset)} val")
    print(f"  Batches:  {len(train_loader)} train  /  {len(val_loader)} val")

    # ── Model ─────────────────────────────────────────────────────
    model = TrayModel(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model:    {cfg.model.backbone}  ({n_params:,} params, {n_trainable:,} trainable)")

    # ── Train ─────────────────────────────────────────────────────
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()