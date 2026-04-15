"""Tests for src.config – loading, defaults, and overrides."""

import yaml

from src.config import load_config, Config


def test_defaults():
    """Config() with no args should have sensible defaults."""
    cfg = Config()
    assert cfg.training.epochs == 30
    assert cfg.model.backbone == "resnet50"
    assert cfg.data.image_size == 640


def test_load_yaml(tmp_path):
    """Load a minimal YAML and verify values propagate."""
    yaml_content = {
        "training": {"epochs": 5, "batch_size": 2},
        "model": {"backbone": "resnet18"},
    }
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(yaml_content))

    cfg = load_config(p)
    assert cfg.training.epochs == 5
    assert cfg.training.batch_size == 2
    assert cfg.model.backbone == "resnet18"
    assert cfg.training.lr == 0.001  # default preserved


def test_cli_overrides(tmp_path):
    """Flat CLI overrides should be placed into the correct sub-config."""
    p = tmp_path / "base.yaml"
    p.write_text(yaml.dump({"training": {"epochs": 10}}))

    cfg = load_config(p, overrides={"epochs": 3, "lr": 0.0005})
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 0.0005


def test_to_dict():
    """Config.to_dict() should return a nested dict."""
    cfg = Config()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["training"]["epochs"] == 30