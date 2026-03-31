"""
pipeline.py – Orchestrates the full tray-to-nutrition pipeline.

    image  →  detect  →  estimate portions  →  nutrition lookup  →  JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from detector import FoodDetector
from portion import estimate_grams
from nutrition import lookup, NutritionEstimate


class TrayPipeline:
    """Stateless pipeline.  Instantiate once (loads model), call many times."""

    def __init__(self):
        self.detector = FoodDetector()

    # ── public API ────────────────────────────────────────────────
    def analyze(self, image_path: str | Path) -> dict:
        """Run full analysis on one tray image.  Returns a dict ready
        for json.dumps().
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        detections = self.detector.detect(img)

        items: list[dict] = []
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

        for det in detections:
            grams = estimate_grams(det, w, h)
            nutr = lookup(det.label, grams)

            item = nutr.to_dict()
            item["confidence"] = det.confidence
            item["bbox"] = list(det.bbox)
            items.append(item)

            for key in totals:
                totals[key] += getattr(nutr, key.replace("_g", "_g") if "_g" in key else key)

        # round totals
        totals = {k: round(v, 1) for k, v in totals.items()}

        return {
            "image": str(image_path),
            "image_size": {"width": w, "height": h},
            "items_detected": len(items),
            "items": items,
            "totals": totals,
        }

    def analyze_to_json(self, image_path: str | Path, pretty: bool = True) -> str:
        """Convenience wrapper that returns a JSON string."""
        result = self.analyze(image_path)
        return json.dumps(result, indent=2 if pretty else None)