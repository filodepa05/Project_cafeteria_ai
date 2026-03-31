"""
detector.py – Wraps YOLOv8 to detect food items in a tray image.

Returns a list of Detection dataclass instances (label, confidence, bbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import FOOD_CLASSES, YOLO_MODEL, CONFIDENCE_THRESHOLD


@dataclass
class Detection:
    """One detected food item."""
    label: str            # e.g. "pizza"
    class_id: int         # COCO class id
    confidence: float     # 0-1
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


class FoodDetector:
    """Loads YOLOv8 once and exposes a simple `detect(image)` method."""

    def __init__(self, model_path: str = YOLO_MODEL):
        self.model = YOLO(model_path)
        self.food_ids = set(FOOD_CLASSES.keys())

    def detect(self, image: str | Path | Image.Image) -> list[Detection]:
        """Run detection on an image path or PIL Image.

        Returns only food-class detections above the confidence threshold.
        """
        results = self.model(image, verbose=False)[0]

        detections: list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in self.food_ids:
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                label=FOOD_CLASSES[cls_id],
                class_id=cls_id,
                confidence=round(conf, 3),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))

        return detections