"""
inference.py – End-to-end inference pipeline.

Two-stage pipeline:
    Stage 1 — YOLO detector  →  bounding boxes per food item
    Stage 2 — per-item       →  classify + nutrition + NLP summary

Graceful fallbacks:
    • No YOLO weights        →  falls back to image-level ResNet classification (MVP mode)
    • No nutrition API       →  falls back to hardcoded USDA table in src/nutrition.py
    • No NLP module          →  summary field is omitted from output

Final JSON output:
    {
        "image": "tray.jpg",
        "mode": "yolo" | "resnet_fallback",
        "items_detected": 3,
        "items": [
            {
                "food": "grilled_chicken",
                "confidence": 0.87,
                "grams": 150.0,
                "calories": 247.5,
                "protein_g": 46.5,
                "carbs_g": 0.0,
                "fat_g": 5.4,
                "bbox": [x1, y1, x2, y2]   # only present in yolo mode
            },
            ...
        ],
        "totals": {"calories": ..., "protein_g": ..., "carbs_g": ..., "fat_g": ...},
        "summary": "NLP text here"           # only present when nlp module available
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torchvision import transforms as T
from PIL import Image

from src.config import Config
from src.dataset import CATEGORIES
from src.models.tray_model import TrayModel
from src.nutrition import estimate_nutrition as estimate_nutrition_fallback
from src.utils.io import resolve_device


# ── Optional module imports (graceful if not yet delivered) ────────

def _try_import_yolo():
    """Returns YOLOFoodDetector class or None if Nicolas hasn't delivered yet."""
    try:
        from src.models.yolo_detector import YOLOFoodDetector
        return YOLOFoodDetector
    except ImportError:
        return None


def _try_import_nutrition_api():
    """Returns nutrition API estimate function or None if JP hasn't delivered yet."""
    try:
        from src.nutrition_api import estimate_nutrition as api_fn
        return api_fn
    except ImportError:
        return None


def _try_import_nlp():
    """Returns NLP summary function or None if Santi hasn't delivered yet."""
    try:
        from src.nlp_summary import generate_summary
        return generate_summary
    except ImportError:
        return None


# ── Main pipeline ──────────────────────────────────────────────────

class TrayInferencePipeline:
    """Full two-stage inference pipeline. Load once, call run() on images.

    Stage 1: YOLO detects food items and returns bounding boxes.
             Falls back to image-level ResNet if YOLO not available.
    Stage 2: Per-item nutrition lookup + NLP summary generation.
    """

    def __init__(self, cfg: Config, checkpoint_path: str | Path):
        self.cfg = cfg
        self.device = resolve_device(cfg.inference.device)

        # ── ResNet classifier (always loaded — backbone + fallback) ───────────
        self.model = TrayModel(cfg.model).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # ── YOLO detector (loaded only if weights path is configured) ─────────
        # Nicolas delivers: src/models/yolo_detector.py + best.pt weights
        # To enable: add `yolo: {weights_path: "path/to/best.pt"}` to base.yaml
        self.yolo = None
        YOLOFoodDetector = _try_import_yolo()
        if YOLOFoodDetector and hasattr(cfg, "yolo") and cfg.yolo.weights_path:
            try:
                self.yolo = YOLOFoodDetector(cfg.yolo.weights_path)
                print("  YOLO detector loaded.", file=sys.stderr)
            except Exception as e:
                print(f"  YOLO load failed ({e}), using ResNet fallback.", file=sys.stderr)

        # ── Optional modules — pipeline works without these ───────────────────
        # JP delivers src/nutrition_api.py  →  replaces hardcoded table
        # Santi delivers src/nlp_summary.py →  adds summary field to output
        self.nutrition_fn = _try_import_nutrition_api() or estimate_nutrition_fallback
        self.nlp_fn = _try_import_nlp()

        if self.nlp_fn is None:
            print("  NLP module not found — summary will be omitted.", file=sys.stderr)

        # ── Image preprocessing (must match training transforms) ──────────────
        self.transform = T.Compose([
            T.Resize((cfg.data.image_size, cfg.data.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # ── Stage 1: YOLO detection ───────────────────────────────────

    def _detect(self, img: Image.Image) -> list[dict] | None:
        """Run YOLO detector on the full tray image.

        Returns list of {label, confidence, bbox: [x1,y1,x2,y2]}
        Returns None if YOLO not available — triggers ResNet fallback.

        Nicolas's YOLOFoodDetector.detect() must return a list of Detection
        objects with: .label (str), .confidence (float), .bbox (x1,y1,x2,y2).
        """
        if self.yolo is None:
            return None

        detections = self.yolo.detect(img)
        return [
            {
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "bbox": list(d.bbox),
            }
            for d in detections
            if d.confidence >= self.cfg.inference.confidence_threshold
        ]

    # ── Stage 1 fallback: ResNet image-level classification ───────

    @torch.no_grad()
    def _classify_image(self, img: Image.Image) -> list[dict]:
        """Image-level multi-label classification via ResNet.

        Used when YOLO is not available. Returns list of
        {label, confidence, grams} for all classes above threshold.
        No bounding boxes — entire image treated as one context.
        """
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        probs = torch.sigmoid(outputs["logits"][0]).cpu()
        grams = outputs["grams"][0, 0].cpu().item()

        return [
            {
                "label": CATEGORIES[i],
                "confidence": round(p.item(), 3),
                "grams": max(30.0, min(round(grams, 1), 400.0)),
            }
            for i, p in enumerate(probs)
            if p.item() >= self.cfg.inference.confidence_threshold
        ]

    # ── Stage 2: nutrition lookup ─────────────────────────────────
    def _get_nutrition(self, label: str, grams: float) -> dict:
        try:
            class_id = CATEGORIES.index(label) if label in CATEGORIES else -1
            result = self.nutrition_fn(class_id, grams)  # now passes 2, not "pizza"
        except (TypeError, KeyError, Exception):
            class_id = CATEGORIES.index(label) if label in CATEGORIES else -1
            result = estimate_nutrition_fallback(class_id, grams)
        return result.to_dict()
    
    # ── Portion estimation from bbox ──────────────────────────────

    @staticmethod
    def _grams_from_bbox(bbox: list[float], img_w: int, img_h: int) -> float:
        """Rough portion estimate based on bounding box area ratio.

        Assumes a full tray holds ~800g of food total.
        Clamps to [30g, 400g] to avoid absurd values.

        TODO (Yago): replace with depth/volume model for better accuracy.
        """
        x1, y1, x2, y2 = bbox
        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        img_area = img_w * img_h
        grams = (bbox_area / img_area) * 800.0
        return round(max(30.0, min(grams, 400.0)), 1)

    # ── Main entry point ──────────────────────────────────────────

    @torch.no_grad()
    def run(self, image_path: str | Path) -> dict:
        """Analyse a single tray image end-to-end.

        Returns a dict ready for json.dumps() with per-item nutrition
        and tray-level totals. Adds NLP summary if Santi's module is loaded.
        """
        img = Image.open(image_path).convert("RGB")

        # Stage 1: detect items
        detections = self._detect(img)
        mode = "yolo" if detections is not None else "resnet_fallback"
        if detections is None:
            detections = self._classify_image(img)

        # Stage 2: nutrition per item
        items = []
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

        for det in detections:
            label = det["label"]

            # Portion size
            if "bbox" in det:
                grams = self._grams_from_bbox(det["bbox"], img.width, img.height)
            else:
                grams = det.get("grams", 150.0)

            nutr = self._get_nutrition(label, grams)

            item = {
                "food": label,
                "confidence": det["confidence"],
                "grams": grams,
                "calories": nutr["calories"],
                "protein_g": nutr["protein_g"],
                "carbs_g": nutr["carbs_g"],
                "fat_g": nutr["fat_g"],
            }
            if "bbox" in det:
                item["bbox"] = det["bbox"]

            items.append(item)
            totals["calories"]  += nutr["calories"]
            totals["protein_g"] += nutr["protein_g"]
            totals["carbs_g"]   += nutr["carbs_g"]
            totals["fat_g"]     += nutr["fat_g"]

        totals = {k: round(v, 1) for k, v in totals.items()}

        result = {
            "image": str(image_path),
            "mode": mode,
            "items_detected": len(items),
            "items": items,
            "totals": totals,
        }

        # NLP summary — added when Santi's module is present
        if self.nlp_fn is not None:
            try:
                result["summary"] = self.nlp_fn(result)
            except Exception as e:
                result["summary"] = f"[summary unavailable: {e}]"

        return result

    def run_to_json(self, image_path: str | Path, pretty: bool = True) -> str:
        result = self.run(image_path)
        return json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)