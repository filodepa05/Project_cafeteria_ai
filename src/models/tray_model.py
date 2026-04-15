"""
tray_model.py – Multi-task model combining detection + portion estimation.

Forward pass returns both classification logits and portion predictions
from a shared backbone, enabling joint training with a weighted loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.detector import DetectionHead
from src.models.portion import PortionHead


class TrayModel(nn.Module):
    """Shared-backbone multi-task model.

    ┌─────────┐     ┌──────────────┐
    │  image  │────►│   backbone   │───► features
    └─────────┘     └──────────────┘        │
                                       ┌────┴────┐
                                       ▼         ▼
                                  ┌─────────┐ ┌─────────┐
                                  │ cls head│ │ portion  │
                                  │ (logits)│ │ (grams)  │
                                  └─────────┘ └─────────┘
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()

        self.detection = DetectionHead(
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
        )
        self.portion = PortionHead(
            feat_dim=self.detection.feat_dim,
            hidden=cfg.portion_hidden,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) image batch

        Returns
        -------
        dict with:
            "logits"  : (B, num_classes) – class logits
            "grams"   : (B, 1)           – estimated portion weight
            "features": (B, feat_dim)    – intermediate features (for debugging)
        """
        logits, features = self.detection(x)
        grams = self.portion(features)

        return {
            "logits": logits,
            "grams": grams,
            "features": features,
        }