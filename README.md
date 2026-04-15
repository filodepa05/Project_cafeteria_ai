# Smart Tray — Cafeteria Tray Nutritional Analysis

Detects food items on a cafeteria tray image, estimates portion sizes, and
outputs calorie + macronutrient breakdowns as structured JSON.

## Architecture

```
image ──► detection head ──► per-item bboxes
               │
               ▼
         portion head ──► estimated grams
               │
               ▼
        nutrition lookup ──► { calories, protein, carbs, fat }
               │
               ▼
          structured JSON
```

## Repository Structure

```
smart-tray/
├── train.py                  # Training entry point
├── infer.py                  # Inference entry point
├── configs/
│   ├── base.yaml             # Default hyperparameters
│   └── experiment/
│       └── debug.yaml        # Fast debug run (2 epochs, tiny batch)
├── src/
│   ├── config.py             # Dataclass-based configuration
│   ├── dataset.py            # TrayDataset (detection + portion labels)
│   ├── models/
│   │   ├── detector.py       # Detection backbone + head
│   │   ├── portion.py        # Portion regression head
│   │   └── tray_model.py     # Combined multi-task model
│   ├── trainer.py            # Training loop with validation
│   ├── inference.py          # End-to-end inference pipeline
│   ├── nutrition.py          # Food → calorie/macro lookup
│   └── utils/
│       ├── io.py             # File I/O helpers
│       ├── viz.py            # Visualization (bbox overlay, charts)
│       └── metrics.py        # mAP, IoU, portion error metrics
├── tests/                    # Unit tests
├── data/                     # ← put datasets here (gitignored)
└── requirements.txt
```

## Quick Start

```bash
# 1 — Install
pip install -r requirements.txt

# 2 — Sanity check with synthetic data (no real images needed)
python train.py --config configs/experiment/debug.yaml

# 3 — Inference on a single image
python infer.py path/to/tray.jpg --output result.json

# 4 — Run tests
pytest tests/ -v
```

## Configuration

All hyperparameters live in YAML files under `configs/`.  Any value can also
be overridden from the CLI:

```bash
python train.py --config configs/base.yaml --epochs 50 --lr 0.0003
```

## Data Format

The dataset expects this layout:

```
data/
├── images/
│   ├── tray_0001.jpg
│   └── ...
└── annotations.json      # COCO-format with extra "portion_grams" field
```

See `src/dataset.py` for the schema.