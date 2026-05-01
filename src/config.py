"""
config.py – Typed configuration via nested dataclasses.

Load order:  base.yaml  ←  experiment yaml  ←  CLI overrides
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── Sub-configs ───────────────────────────────────────────────────

@dataclass
class DataConfig:
    root: str = "data"
    image_size: int = 640
    train_split: float = 0.8
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 43
    portion_hidden: int = 256


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    step_size: int = 10
    gamma: float = 0.1
    detection_loss_weight: float = 1.0
    portion_loss_weight: float = 0.5


@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.30
    nms_iou_threshold: float = 0.50
    device: str = "auto"


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_every: int = 5
    keep_top_k: int = 3


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_every_n_steps: int = 20
    use_rich: bool = True


@dataclass
class DebugConfig:
    use_synthetic: bool = False
    synthetic_samples: int = 32


@dataclass
class YoloConfig:
    weights_path: str = "src/weights/yolo_best.pt"


@dataclass
class NutritionConfig:
    usda_api_key: str | None = None
    cache_path: str = "data/nutrition_cache.json"
    use_api: bool = True


# ── Root config ───────────────────────────────────────────────────

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    nutrition: NutritionConfig = field(default_factory=NutritionConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    _debug: DebugConfig = field(default_factory=DebugConfig)

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False))


# ── Loader ────────────────────────────────────────────────────────

_SUB_MAP: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "inference": InferenceConfig,
    "checkpoint": CheckpointConfig,
    "logging": LoggingConfig,
    "nutrition": NutritionConfig,
    "yolo": YoloConfig,
    "_debug": DebugConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(yaml_path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and apply optional CLI overrides.

    Parameters
    ----------
    yaml_path : path to YAML file
    overrides : flat dict of CLI overrides, e.g. {"epochs": 5, "lr": 0.0003}
                Keys are matched to the *leaf* field name across all sub-configs.

    Returns
    -------
    Config dataclass instance, fully populated.
    """
    raw: dict = yaml.safe_load(Path(yaml_path).read_text(encoding='utf-8')) or {}

    # Apply flat CLI overrides → nested structure
    if overrides:
        for key, value in overrides.items():
            placed = False
            for section_name, section_cls in _SUB_MAP.items():
                if key in section_cls.__dataclass_fields__:
                    raw.setdefault(section_name, {})[key] = value
                    placed = True
                    break
            if not placed:
                raise ValueError(f"Unknown config key: {key}")

    # Build sub-config dataclasses
    kwargs: dict[str, Any] = {}
    for section_name, section_cls in _SUB_MAP.items():
        section_data = raw.get(section_name, {})
        valid_fields = set(section_cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in section_data.items() if k in valid_fields}
        kwargs[section_name] = section_cls(**filtered)

    return Config(**kwargs)