"""
viz.py – Visualisation utilities for training and evaluation plots.

Usage:
    from src.utils.viz import plot_training_curves
    plot_training_curves(history, save_path="plots/loss_curves.png")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works on all systems
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: dict,
    save_path: str | Path = "plots/loss_curves.png",
    dpi: int = 150,
) -> None:
    """Plot training and validation loss curves.

    Generates a single plot with:
      - Main panel: total train loss (solid blue) vs val loss (dashed orange)
      - Inset panel: BCE classification loss and MSE portion loss separately

    Parameters
    ----------
    history : dict with keys:
        "train_total", "val_total",
        "train_cls",   "val_cls",
        "train_portion", "val_portion"
    save_path : where to save the figure
    dpi : export resolution (default 150)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_total"]) + 1)

    fig, ax_main = plt.subplots(figsize=(10, 6))
    fig.suptitle("ResNet18 Training & Validation Loss", fontsize=14, fontweight="bold")

    # ── Main panel: total loss ─────────────────────────────────────
    ax_main.plot(epochs, history["train_total"],
                 label="Train Loss", color="steelblue",
                 linewidth=2.5, linestyle="-")
    ax_main.plot(epochs, history["val_total"],
                 label="Val Loss", color="darkorange",
                 linewidth=2.5, linestyle="--")
    ax_main.set_xlabel("Epoch", fontsize=12)
    ax_main.set_ylabel("Total Loss", fontsize=12)
    ax_main.legend(fontsize=11, loc="upper right")
    ax_main.grid(True, alpha=0.3)

    # Mark best val loss
    best_epoch = int(np.argmin(history["val_total"])) + 1
    best_val   = min(history["val_total"])
    ax_main.axvline(x=best_epoch, color="gray", linestyle=":", alpha=0.7)
    ax_main.annotate(
        f"Best epoch {best_epoch}\nloss={best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 0.5, best_val * 1.05),
        fontsize=9,
        color="gray",
    )

    # ── Inset panel: BCE + MSE separately ─────────────────────────
    # Position: top-right corner inside the main plot
    ax_inset = ax_main.inset_axes([0.55, 0.45, 0.42, 0.45])

    ax_inset.plot(epochs, history["train_cls"], label="Train BCE",
                  color="steelblue", linewidth=1.5, linestyle="-")
    ax_inset.plot(epochs, history["val_cls"], label="Val BCE",
                  color="darkorange", linewidth=1.5, linestyle="--")
    ax_inset.plot(epochs, history["train_portion"], label="Train MSE",
                  color="mediumseagreen", linewidth=1.5, linestyle="-")
    ax_inset.plot(epochs, history["val_portion"], label="Val MSE",
                  color="tomato", linewidth=1.5, linestyle="--")

    ax_inset.set_title("BCE & MSE separately", fontsize=9)
    ax_inset.set_xlabel("Epoch", fontsize=8)
    ax_inset.legend(fontsize=7, loc="upper right")
    ax_inset.grid(True, alpha=0.3)
    ax_inset.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")