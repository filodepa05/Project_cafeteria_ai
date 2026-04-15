"""
metrics.py – Evaluation metrics for the tray model.
"""

from __future__ import annotations

import torch


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Multi-label accuracy: fraction of (sample, class) predictions correct.

    Parameters
    ----------
    logits  : (B, C) raw logits
    targets : (B, C) multi-hot ground truth (0 or 1)

    Returns
    -------
    Accuracy in [0, 1].
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    correct = (preds == targets).float().mean()
    return correct.item()


def compute_portion_mae(pred_grams: torch.Tensor, true_grams: torch.Tensor) -> float:
    """Mean absolute error for portion estimation.

    Parameters
    ----------
    pred_grams : (B,) or (B,1)
    true_grams : (B,)

    Returns
    -------
    MAE in grams.
    """
    return torch.abs(pred_grams.squeeze() - true_grams).mean().item()


def per_class_precision_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[int, dict[str, float]]:
    """Compute per-class precision and recall.

    Returns
    -------
    {class_id: {"precision": float, "recall": float}}
    """
    preds = (torch.sigmoid(logits) >= threshold).float()
    C = logits.shape[1]
    results = {}

    for c in range(C):
        tp = ((preds[:, c] == 1) & (targets[:, c] == 1)).sum().float()
        fp = ((preds[:, c] == 1) & (targets[:, c] == 0)).sum().float()
        fn = ((preds[:, c] == 0) & (targets[:, c] == 1)).sum().float()

        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        results[c] = {"precision": round(precision, 4), "recall": round(recall, 4)}

    return results