"""
trainer.py – Training loop for the TrayModel.

Handles:
  • Multi-task loss (classification BCE + portion MSE)
  • LR scheduling (cosine / step)
  • Checkpoint saving (top-K by val loss)
  • Rich progress bars
  • Training curves via src/utils/viz.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.config import Config
from src.models.tray_model import TrayModel
from src.utils.io import resolve_device
from src.utils.viz import plot_training_curves

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class Trainer:
    """End-to-end trainer for the multi-task TrayModel."""

    def __init__(self, model: TrayModel, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.inference.device)
        self.model = model.to(self.device)

        # ── Losses ────────────────────────────────────────────────
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.portion_loss_fn = nn.MSELoss()

        # ── Optimizer ─────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        # ── Scheduler ─────────────────────────────────────────────
        if cfg.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.epochs)
        elif cfg.training.scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
        else:
            self.scheduler = None

        # ── Checkpointing ─────────────────────────────────────────
        self.ckpt_dir = Path(cfg.checkpoint.save_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_losses: list[tuple[float, Path]] = []

        # ── Logging ───────────────────────────────────────────────
        self.console = Console() if HAS_RICH else None
        self.log_dir = Path(cfg.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ── Plots directory ───────────────────────────────────────
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # ── History (for curves) ──────────────────────────────────
        self.history = {
            "train_total": [], "train_cls": [], "train_portion": [],
            "val_total":   [], "val_cls":   [], "val_portion":   [],
            "lr": [],
        }

    # ── Loss computation ──────────────────────────────────────────

    def _compute_loss(self, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        grams_pred = outputs["grams"].squeeze(-1)
        B, C = logits.shape

        targets = torch.zeros(B, C, device=logits.device)
        for i, lbls in enumerate(batch["labels"]):
            for lbl in lbls:
                if lbl < C:
                    targets[i, lbl] = 1.0

        portion_targets = torch.zeros(B, device=logits.device)
        for i, p in enumerate(batch["portions"]):
            portion_targets[i] = p.mean() if len(p) > 0 else 0.0

        cls_loss = self.cls_loss_fn(logits, targets)
        portion_loss = self.portion_loss_fn(grams_pred, portion_targets)

        w_cls = self.cfg.training.detection_loss_weight
        w_por = self.cfg.training.portion_loss_weight
        total = w_cls * cls_loss + w_por * portion_loss

        return {"total": total, "cls": cls_loss, "portion": portion_loss}

    # ── Train one epoch ───────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running = {"total": 0.0, "cls": 0.0, "portion": 0.0}
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            outputs = self.model(images)
            losses = self._compute_loss(outputs, batch)

            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ── Validate ──────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        running = {"total": 0.0, "cls": 0.0, "portion": 0.0}
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            outputs = self.model(images)
            losses = self._compute_loss(outputs, batch)

            for k in running:
                running[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ── Checkpoint management ─────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.ckpt_dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)

        self._best_losses.append((val_loss, path))
        self._best_losses.sort(key=lambda x: x[0])

        while len(self._best_losses) > self.cfg.checkpoint.keep_top_k:
            _, old_path = self._best_losses.pop()
            old_path.unlink(missing_ok=True)

    # ── Main training loop ────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.cfg.training.epochs
        print(f"\n{'═' * 60}")
        print(f"  Training TrayModel on {self.device}  —  {epochs} epochs")
        print(f"{'═' * 60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(train_loader)
            val_metrics   = self._val_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Save to history
            self.history["train_total"].append(train_metrics["total"])
            self.history["train_cls"].append(train_metrics["cls"])
            self.history["train_portion"].append(train_metrics["portion"])
            self.history["val_total"].append(val_metrics["total"])
            self.history["val_cls"].append(val_metrics["cls"])
            self.history["val_portion"].append(val_metrics["portion"])
            self.history["lr"].append(lr)

            print(
                f"  Epoch {epoch:3d}/{epochs}  │  "
                f"train_loss {train_metrics['total']:.4f}  "
                f"(cls {train_metrics['cls']:.4f}  por {train_metrics['portion']:.4f})  │  "
                f"val_loss {val_metrics['total']:.4f}  │  "
                f"lr {lr:.6f}  │  {elapsed:.1f}s"
            )

            if epoch % self.cfg.checkpoint.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["total"])

        # Always save the final model
        self._save_checkpoint(epochs, val_metrics["total"])
        print(f"\n  Training complete.  Best val loss: {self._best_losses[0][0]:.4f}")
        print(f"  Checkpoints saved to: {self.ckpt_dir}/\n")

        # ── Generate plots ────────────────────────────────────────
        print("  Generating report plots...")
        plot_training_curves(
            history=self.history,
            save_path=self.plots_dir / "loss_curves.png",
            dpi=150,
        )
        print(f"\n  All plots saved to: {self.plots_dir}/")
        print("  └── loss_curves.png")