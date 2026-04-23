"""Tests for the TrayModel – shapes, gradients, serialisation."""

import torch
from src.models.tray_model import TrayModel


def test_forward_shapes(model):
    """Model output tensors should have the expected shapes."""
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out["logits"].shape == (2, 43)
    assert out["grams"].shape == (2, 1)
    assert out["features"].shape[0] == 2


def test_gradients_flow(model):
    """Backward pass should produce gradients on all parameters."""
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = out["logits"].sum() + out["grams"].sum()
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_checkpoint_roundtrip(model, tmp_path):
    """Save and reload should produce identical outputs."""
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        original = model(x)["logits"]

    ckpt_path = tmp_path / "test.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    from src.config import ModelConfig
    loaded = TrayModel(ModelConfig(backbone="resnet18", pretrained=False, num_classes=43, portion_hidden=32))
    ckpt = torch.load(ckpt_path, weights_only=False)
    loaded.load_state_dict(ckpt["model_state_dict"])
    loaded.eval()

    with torch.no_grad():
        reloaded = loaded(x)["logits"]

    assert torch.allclose(original, reloaded, atol=1e-6)