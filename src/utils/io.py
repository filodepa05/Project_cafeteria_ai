"""
io.py – File I/O and device utilities.
"""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_device(device_str: str = "auto") -> torch.device:
    """Return the best available device.

    'auto' → cuda if available, then mps (Apple Silicon), then cpu.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load a .pt checkpoint dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def find_best_checkpoint(ckpt_dir: str | Path) -> Path:
    """Find the checkpoint with the lowest val loss based on filename convention.

    Expected format: epoch_NNN_loss_X.XXXX.pt
    """
    ckpt_dir = Path(ckpt_dir)
    candidates = sorted(ckpt_dir.glob("epoch_*_loss_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def _parse_loss(p: Path) -> float:
        try:
            return float(p.stem.split("loss_")[1])
        except (IndexError, ValueError):
            return float("inf")

    return min(candidates, key=_parse_loss)