"""
conftest.py – Shared fixtures for pytest.
"""

import pytest
import torch

from src.config import load_config, Config, ModelConfig, DataConfig, DebugConfig
from src.dataset import SyntheticTrayDataset, collate_fn
from src.models.tray_model import TrayModel


@pytest.fixture
def debug_config(tmp_path) -> Config:
    """A minimal config for fast unit tests."""
    return Config(
        data=DataConfig(image_size=64, num_workers=0, pin_memory=False),
        model=ModelConfig(backbone="resnet18", pretrained=False, num_classes=10, portion_hidden=32),
        _debug=DebugConfig(use_synthetic=True, synthetic_samples=8),
    )


@pytest.fixture
def model(debug_config) -> TrayModel:
    return TrayModel(debug_config.model)


@pytest.fixture
def synthetic_batch(debug_config) -> dict:
    """A single collated batch of 4 synthetic samples."""
    ds = SyntheticTrayDataset(n_samples=4, image_size=64, num_classes=10)
    samples = [ds[i] for i in range(4)]
    return collate_fn(samples)