"""
inference.py – End-to-end inference pipeline.

    image  →  TrayModel  →  predicted classes + grams  →  nutrition lookup  →  JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import transforms as T
from PIL import Image

from src.config import Config
from src.dataset import CATEGORIES
from src.models.tray_model import TrayModel
from src.nutrition import estimate_nutrition
from src.utils.io import resolve_device


class TrayInferencePipeline:
    """Stateless inference wrapper.  Load once, call `run()` on images."""

    def __init__(self, cfg: Config, checkpoint_path: str | Path):
        self.cfg = cfg
        self.device = resolve_device(cfg.inference.device)

        # Load model
        self.model = TrayModel(cfg.model).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Preprocessing (must match training)
        self.transform = T.Compose([
            T.Resize((cfg.data.image_size, cfg.data.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def run(self, image_path: str | Path) -> dict:
        """Analyse a single tray image.

        Returns
        -------
        dict ready for json.dumps() with per-item nutrition + totals.
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        outputs = self.model(tensor)
        probs = torch.sigmoid(outputs["logits"][0]).cpu()           # (C,)
        # NOTE: grams is currently an image-level prediction (one value per tray).
        # Per-item portion estimation is a planned milestone (see roadmap).
        # All detected items receive the same gram estimate for now.
        grams = outputs["grams"][0, 0].cpu().item()                 # scalar

        threshold = self.cfg.inference.confidence_threshold

        items = []
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

        for cls_id, p in enumerate(probs):
            if p.item() < threshold:
                continue

            nutr = estimate_nutrition(cls_id, grams)
            item = nutr.to_dict()
            item["confidence"] = round(p.item(), 3)
            items.append(item)

            totals["calories"] += nutr.calories
            totals["protein_g"] += nutr.protein_g
            totals["carbs_g"] += nutr.carbs_g
            totals["fat_g"] += nutr.fat_g

        totals = {k: round(v, 1) for k, v in totals.items()}

        return {
            "image": str(image_path),
            "items_detected": len(items),
            "items": items,
            "totals": totals,
        }

    def run_to_json(self, image_path: str | Path, pretty: bool = True) -> str:
        result = self.run(image_path)
        return json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)