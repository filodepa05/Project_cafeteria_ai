"""
viz.py – Visualization utilities for debugging and reporting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_predictions(
    image_path: str | Path,
    items: list[dict],
    save_path: str | Path | None = None,
) -> Image.Image:
    """Draw predicted food labels and confidence on an image.

    Parameters
    ----------
    image_path : path to the original image
    items      : list of dicts from the inference pipeline (food, confidence, grams, etc.)
    save_path  : optional path to save the annotated image

    Returns
    -------
    PIL Image with annotations.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Build label text for each detected item
    y_offset = 10
    for item in items:
        label = f"{item['food']}  {item['confidence']:.0%}  ~{item['grams']:.0f}g  {item['calories']:.0f}kcal"
        draw.text((10, y_offset), label, fill="lime")
        y_offset += 20

    if save_path:
        img.save(save_path)
    return img


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str | Path = "training_curves.png",
) -> None:
    """Plot and save train/val loss curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training curves to {save_path}")