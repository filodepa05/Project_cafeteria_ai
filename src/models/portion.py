"""
portion.py – Portion size regression head.

Takes the shared feature vector from DetectionHead and regresses
an estimated portion weight in grams.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PortionHead(nn.Module):
    """3-layer MLP: feature_vector → estimated grams.

    Architecture: feat_dim → hidden → hidden//2 → 1
    """

    def __init__(self, feat_dim: int = 2048, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, feat_dim) – pooled features from DetectionHead

        Returns
        -------
        grams : (B, 1) – predicted portion weight
        """
        return self.mlp(features)