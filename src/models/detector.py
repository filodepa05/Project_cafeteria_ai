"""
detector.py – Feature backbone + food classification head.

MVP approach:
  ResNet backbone → global average pool → FC → per-class logits

This gives us image-level multi-label food classification.  A later
milestone swaps this for torchvision Faster R-CNN to get real bounding
boxes, but the current version lets us train end-to-end TODAY.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# Map config string → torchvision constructor + feature dim
_BACKBONES: dict[str, tuple] = {
    "resnet18":          (models.resnet18,          512),
    "resnet50":          (models.resnet50,          2048),
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
}


class DetectionHead(nn.Module):
    """Backbone → feature vector → food class logits.

    Attributes
    ----------
    backbone : nn.Module   – pretrained feature extractor (everything before FC)
    feat_dim : int         – dimensionality of the feature vector
    classifier : nn.Module – linear head  →  (batch, num_classes)
    """

    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, num_classes: int = 10):
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONES)}")

        factory, self.feat_dim = _BACKBONES[backbone_name]

        if backbone_name.startswith("resnet"):
            weights = "IMAGENET1K_V1" if pretrained else None
            full = factory(weights=weights)
            # Strip the final FC — keep everything up to avgpool
            self.backbone = nn.Sequential(*list(full.children())[:-1])  # → (B, feat_dim, 1, 1)
        else:
            weights = "IMAGENET1K_V1" if pretrained else None
            full = factory(weights=weights)
            self.backbone = full.features
            self.backbone.add_module("pool", nn.AdaptiveAvgPool2d(1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) image tensor

        Returns
        -------
        logits   : (B, num_classes)  – raw logits for BCE loss
        features : (B, feat_dim)     – pooled features for the portion head
        """
        feats = self.backbone(x)            # (B, feat_dim, 1, 1)
        feats_flat = feats.flatten(1)       # (B, feat_dim)
        logits = self.classifier(feats_flat)
        return logits, feats_flat