"""Tests for nutrition lookup and scaling."""

from src.nutrition import estimate_nutrition


def test_banana_100g():
    """100g of banana should return exact per-100g values."""
    n = estimate_nutrition(class_id=0, grams=100.0)
    assert n.food == "banana"
    assert n.calories == 89.0
    assert n.protein_g == 1.1


def test_scaling_200g():
    """200g should double the 100g values."""
    n = estimate_nutrition(class_id=0, grams=200.0)
    assert n.calories == 178.0
    assert n.protein_g == 2.2


def test_unknown_class():
    """Out-of-range class id should return zeroes without crashing."""
    n = estimate_nutrition(class_id=999, grams=150.0)
    assert n.food == "unknown"
    assert n.calories == 0


def test_zero_grams():
    """Zero grams should return zero nutrition."""
    n = estimate_nutrition(class_id=7, grams=0.0)
    assert n.calories == 0.0
    assert n.fat_g == 0.0


def test_to_dict_keys():
    """to_dict should produce the expected key set."""
    n = estimate_nutrition(class_id=3, grams=150.0)
    d = n.to_dict()
    assert set(d.keys()) == {"food", "grams", "calories", "protein_g", "carbs_g", "fat_g"}