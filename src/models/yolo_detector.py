"""
yolo_detector.py — YOLOv8 food detector.

Replaces the image-level classification approach in detector.py with real
bounding-box detection. Used by the inference pipeline to localise individual
food items on a tray image; each cropped region is then passed to the ResNet
classifier (detector.py) for food-class identification.

Trained on a merged dataset:
  - UNIMIB2016 canteen tray photos (961 images, ~2150 boxes, generic food)
  - IE cafeteria photos labelled in Roboflow (134 images, ~440 boxes,
    43 fine-grained classes)
Total: 44 classes (43 fine-grained + 'food_generic' fallback).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    """One detected food item on a tray."""
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates


class YOLOFoodDetector:
    """Wraps a fine-tuned YOLOv8 model with a clean detection interface.

    Parameters
    ----------
    weights_path : str
        Path to the .pt file produced by `yolo detect train ...`.
    conf_threshold : float, default 0.25
        Minimum confidence to report a detection. Lower = more boxes,
        more false positives. The default matches Ultralytics' inference default.
    iou_threshold : float, default 0.45
        IoU threshold for non-max suppression. Lower = more aggressive box
        merging when boxes overlap.
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.model = YOLO(weights_path)
        self.names = self.model.names  # {0: 'pasta', 1: 'rice', ..., 43: 'food_generic'}
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: Union[str, Image.Image, np.ndarray]) -> list[Detection]:
        """Run detection on a single image.

        Parameters
        ----------
        image : str | PIL.Image | np.ndarray
            File path, PIL image, or numpy array (H, W, 3) in RGB.

        Returns
        -------
        list[Detection]
            One Detection per food item found. Empty list if nothing detected.
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=self.names[cls_id],
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
