"""
evaluate.py – Compute validation metrics for the report table.

Usage:
    python evaluate.py --checkpoint checkpoints/epoch_018_loss_0.0906.pt

Outputs:
    - Multi-label accuracy (cls head) at threshold 0.5
    - Macro-averaged F1 (cls head) over 43 classes
    - Portion MAE (grams)
    - Portion RMSE (grams)
    - Calories MAE (kcal)
    - Val loss (combined)
"""

from __future__ import annotations

import argparse
import torch
import numpy as np
from pathlib import Path

from src.config import load_config
from src.dataset import build_dataloaders, CATEGORIES
from src.models.tray_model import TrayModel
from src.utils.io import resolve_device

# Calories per 100g for each of the 43 classes (approximate reference values)
KCAL_PER_100G = {
    "pasta": 131, "rice": 130, "pizza": 266, "bread": 265, "fries": 312,
    "couscous": 112, "potatoes": 77, "wrap_sandwich": 220,
    "grilled_chicken": 165, "fried_chicken": 260, "chicken_stew": 150,
    "turkey": 135, "grilled_beef": 250, "beef_stew": 180, "meatballs": 220,
    "grilled_pork": 242, "pork_ribs": 292, "salmon": 208, "hake": 90,
    "tuna": 144, "cod": 82, "grilled_fish": 100, "fried_fish": 200,
    "eggs": 155, "lentils": 116, "chickpeas": 164, "salad": 20,
    "soup_cream": 60, "grilled_vegetables": 45, "sauteed_vegetables": 55,
    "broccoli": 35, "stuffed_peppers": 90, "poke_bowl": 150,
    "lasagne": 180, "fresh_fruit": 50, "fruit_salad": 55, "yogurt": 61,
    "cake_pastry": 380, "ice_cream_sorbet": 200, "juice_drink": 45,
    "rotisserie_chicken": 190, "fried_potatoes": 300, "baked_potatoes": 93,
}
KCAL_LIST = [KCAL_PER_100G[c] for c in CATEGORIES]


def evaluate(checkpoint_path: str, config_path: str = "configs/base.yaml"):
    # ── Setup ──────────────────────────────────────────────────────
    cfg = load_config(config_path)
    device = resolve_device(cfg.inference.device)

    # Load model
    model = TrayModel(cfg.model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n  Loaded checkpoint: {checkpoint_path}")
    print(f"  Checkpoint epoch:  {ckpt.get('epoch', '?')}")
    print(f"  Checkpoint val loss: {ckpt.get('val_loss', '?'):.4f}\n")

    # Load val data
    _, val_loader = build_dataloaders(cfg)

    # ── Collect predictions ────────────────────────────────────────
    all_logits   = []
    all_targets  = []
    all_grams_pred = []
    all_grams_true = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            logits = outputs["logits"].cpu()       # (B, C)
            grams  = outputs["grams"].squeeze(-1).cpu()  # (B,)

            B, C = logits.shape

            # Build multi-hot targets
            targets = torch.zeros(B, C)
            for i, lbls in enumerate(batch["labels"]):
                for lbl in lbls:
                    if lbl < C:
                        targets[i, lbl] = 1.0

            # Average portion per image
            portion_targets = torch.zeros(B)
            for i, p in enumerate(batch["portions"]):
                portion_targets[i] = p.mean() if len(p) > 0 else 150.0

            all_logits.append(logits)
            all_targets.append(targets)
            all_grams_pred.append(grams)
            all_grams_true.append(portion_targets)

    all_logits    = torch.cat(all_logits,    dim=0).numpy()   # (N, C)
    all_targets   = torch.cat(all_targets,   dim=0).numpy()
    all_grams_pred = torch.cat(all_grams_pred, dim=0).numpy()
    all_grams_true = torch.cat(all_grams_true, dim=0).numpy()

    # ── Classification metrics ─────────────────────────────────────
    probs = 1 / (1 + np.exp(-all_logits))   # sigmoid
    preds = (probs >= 0.5).astype(int)

    # Multi-label accuracy: fraction of (sample, class) pairs correct
    multilabel_acc = (preds == all_targets.astype(int)).mean() * 100

    # Macro F1 over 43 classes
    f1_per_class = []
    for c in range(all_targets.shape[1]):
        tp = ((preds[:, c] == 1) & (all_targets[:, c] == 1)).sum()
        fp = ((preds[:, c] == 1) & (all_targets[:, c] == 0)).sum()
        fn = ((preds[:, c] == 0) & (all_targets[:, c] == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)
    macro_f1 = np.mean(f1_per_class)

    # ── Portion metrics ────────────────────────────────────────────
    mae  = np.abs(all_grams_pred - all_grams_true).mean()
    rmse = np.sqrt(((all_grams_pred - all_grams_true) ** 2).mean())

    # ── Calories MAE ──────────────────────────────────────────────
    # Use predicted top-1 class to get kcal/100g reference
    top1_class = np.argmax(all_logits, axis=1)
    kcal_ref   = np.array([KCAL_LIST[c] for c in top1_class])
    cal_pred   = all_grams_pred * kcal_ref / 100.0
    cal_true   = all_grams_true * kcal_ref / 100.0
    cal_mae    = np.abs(cal_pred - cal_true).mean()

    # ── Print results ──────────────────────────────────────────────
    print("=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Multi-label accuracy (cls head): {multilabel_acc:.1f}%")
    print(f"  Macro-averaged F1   (cls head):  {macro_f1:.4f}")
    print(f"  Portion MAE  (grams):            {mae:.1f} g")
    print(f"  Portion RMSE (grams):            {rmse:.1f} g")
    print(f"  Calories MAE (kcal):             {cal_mae:.1f} kcal")
    print(f"  Val loss (combined):             {ckpt.get('val_loss', '?'):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)
