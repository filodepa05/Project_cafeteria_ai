"""
portion.py – Estimates portion weight (grams) from bounding-box size.

Strategy (MVP):
  1. Compute what fraction of the image the bbox occupies.
  2. Compare that fraction to a reference fraction (config.REFERENCE_BBOX_FRACTION).
  3. Scale the food's reference weight proportionally.

This is a rough proxy.  Later milestones swap in depth estimation
or learned volume models.
"""

from __future__ import annotations

from config import REFERENCE_BBOX_FRACTION, REFERENCE_WEIGHT_G
from detector import Detection


def estimate_grams(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> float:
    """Return estimated weight in grams for a single detection."""

    x1, y1, x2, y2 = detection.bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height

    bbox_fraction = bbox_area / image_area if image_area > 0 else 0.0
    scale = bbox_fraction / REFERENCE_BBOX_FRACTION

    ref_weight = REFERENCE_WEIGHT_G.get(detection.label, 150.0)
    return round(ref_weight * scale, 1)