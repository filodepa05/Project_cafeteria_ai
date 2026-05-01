# Smart Tray — Automated Cafeteria Tray Nutritional Analysis

## Overview

Smart Tray is an end-to-end computer vision system that analyzes cafeteria tray images to:

* Detect food items
* Estimate portion sizes (grams)
* Compute nutritional information (calories, protein, carbs, fat)

The system integrates object detection, regression, and external nutrition data into a unified inference pipeline that outputs structured JSON.

---

## System Architecture

```
Image
  │
  ▼
Object Detection (YOLO-based)
  │
  ▼
Bounding Boxes + Class Labels
  │
  ▼
Portion Estimation (Regression Head)
  │
  ▼
Estimated Grams per Item
  │
  ▼
Nutrition Lookup (USDA API / Fallback)
  │
  ▼
Structured Output (JSON)
```

---

## Key Features

* Multi-task learning: detection + portion estimation
* Real-time inference capability
* Robust fallback nutrition system (works without API)
* Modular and extensible architecture
* Config-driven experimentation
* Clean JSON outputs for downstream integration

---

## Repository Structure

```
smart-tray/
├── train.py                  # Training entry point
├── infer.py                  # Inference entry point
├── demo.py                   # Interactive demo (Gradio UI)
├── configs/                  # YAML configuration files
│   ├── base.yaml
│   └── experiment/
├── src/
│   ├── config.py             # Configuration handling
│   ├── dataset.py            # Dataset loader (COCO + portions)
│   ├── models/
│   │   ├── detector.py       # Detection model (YOLO backbone)
│   │   ├── portion.py        # Portion regression head
│   │   └── tray_model.py     # Combined multi-task model
│   ├── trainer.py            # Training + validation loop
│   ├── inference.py          # Full inference pipeline
│   ├── nutrition.py          # Nutrition lookup (API + fallback)
│   └── utils/
│       ├── io.py
│       ├── viz.py
│       └── metrics.py
├── tests/                    # Unit tests
├── data/                     # Dataset directory (not tracked)
├── checkpoints/              # Model weights
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run Debug Training (Sanity Check)

```bash
python train.py --config configs/experiment/debug.yaml
```

Runs a short training cycle on synthetic/small data to verify pipeline correctness.

---

### 2. Run Inference

```bash
python infer.py examples/tray_1.jpg --auto-checkpoint
```

Output:

* Detected food items
* Portion estimates
* Nutritional breakdown
* JSON summary

---

### 3. Run Demo UI (Optional)

```bash
python demo.py
```

Opens a local web interface for interactive testing.

---

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Configuration System

All hyperparameters are defined in YAML files under `configs/`.

Example override:

```bash
python train.py --config configs/base.yaml --epochs 50 --lr 0.0003
```

---

## Dataset Format

Expected structure:

```
data/
├── images/
│   ├── tray_0001.jpg
│   └── ...
└── annotations.json
```

Annotations follow COCO format with an additional field:

```
"portion_grams": float
```

---

## Nutrition Data

The system supports two modes:

### 1. USDA API (Recommended)

Provides real nutritional values.

Set API key:

```bash
# Windows PowerShell
$env:USDA_API_KEY="YOUR_KEY"
```

### 2. Fallback Mode

Used automatically if API is unavailable.
Ensures system robustness but may reduce accuracy.

---

## Output Format

Example JSON output:

```json
{
  "items_detected": 2,
  "items": [...],
  "totals": {
    "calories": 389.9,
    "protein_g": 34.3,
    "carbs_g": 25.7,
    "fat_g": 16.5
  }
}
```

---

## Evaluation Metrics

* Detection: mAP, IoU
* Portion estimation: MAE / MSE (grams)
* End-to-end: nutritional error vs ground truth

---

## Limitations

* Portion estimation accuracy depends on training data quality
* Food appearance variability may reduce detection accuracy
* Nutrition lookup relies on matching predicted labels to database entries

---

## Future Work

* Improve portion estimation using depth or multi-view inputs
* Expand food category coverage
* Optimize inference latency
* Add meal recommendation system

---

## Conclusion

Smart Tray demonstrates a practical application of computer vision in food analytics by combining detection, regression, and external data integration into a robust, modular pipeline suitable for real-world deployment.
