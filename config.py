"""
config.py – Central configuration for the food-tray estimator MVP.

Holds:
  • COCO class-id → food-name mapping (only food-related classes)
  • Nutrition lookup per 100 g (calories, protein, carbs, fat)
  • Portion-estimation constants
"""

# ── COCO food classes (id → name) ──────────────────────────────────
# YOLOv8 trained on COCO has 80 classes; these are the food ones.
FOOD_CLASSES: dict[int, str] = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot_dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}

# ── Nutrition per 100 g (approximate USDA values) ─────────────────
# Keys must match the values in FOOD_CLASSES above.
NUTRITION_PER_100G: dict[str, dict[str, float]] = {
    "banana":   {"calories": 89,  "protein": 1.1, "carbs": 22.8, "fat": 0.3},
    "apple":    {"calories": 52,  "protein": 0.3, "carbs": 13.8, "fat": 0.2},
    "sandwich": {"calories": 250, "protein": 11.0, "carbs": 28.0, "fat": 10.0},
    "orange":   {"calories": 47,  "protein": 0.9, "carbs": 11.8, "fat": 0.1},
    "broccoli": {"calories": 34,  "protein": 2.8, "carbs": 6.6,  "fat": 0.4},
    "carrot":   {"calories": 41,  "protein": 0.9, "carbs": 9.6,  "fat": 0.2},
    "hot_dog":  {"calories": 290, "protein": 10.0, "carbs": 24.0, "fat": 18.0},
    "pizza":    {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    "donut":    {"calories": 452, "protein": 5.0, "carbs": 51.0, "fat": 25.0},
    "cake":     {"calories": 350, "protein": 4.5, "carbs": 50.0, "fat": 15.0},
}

# ── Typical single-item weight in grams ───────────────────────────
# Used as a baseline; the portion estimator scales from here.
REFERENCE_WEIGHT_G: dict[str, float] = {
    "banana":   120.0,
    "apple":    180.0,
    "sandwich": 200.0,
    "orange":   150.0,
    "broccoli": 150.0,
    "carrot":   80.0,
    "hot_dog":  175.0,
    "pizza":    150.0,   # one slice
    "donut":    60.0,
    "cake":     100.0,   # one slice
}

# ── Portion estimation constants ──────────────────────────────────
# A "reference bbox fraction" is the expected fraction of the image
# area that one standard serving occupies.  We assume a typical tray
# photo where one item takes up roughly 8 % of the frame.
REFERENCE_BBOX_FRACTION = 0.08

# ── Detector settings ─────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # nano model – fast, small
CONFIDENCE_THRESHOLD = 0.30