"""Tests for nutrition lookup and scaling."""

from src.nutrition import estimate_nutrition, CATEGORIES


def test_pasta_100g():
    """100g of pasta should return exact per-100g values."""
    pasta_id = CATEGORIES.index("pasta")
    n = estimate_nutrition(class_id=pasta_id, grams=100.0)
    assert n.food == "pasta"
    assert n.calories == 158.0
    assert n.protein_g == 5.8


def test_scaling_200g():
    """200g should double the 100g values."""
    pasta_id = CATEGORIES.index("pasta")
    n = estimate_nutrition(class_id=pasta_id, grams=200.0)
    assert n.calories == 316.0  # 158 * 2
    assert n.protein_g == 11.6  # 5.8 * 2


def test_unknown_class():
    """Out-of-range class id should return zeroes without crashing."""
    n = estimate_nutrition(class_id=999, grams=150.0)
    assert n.food == "unknown"
    assert n.calories == 0


def test_zero_grams():
    """Zero grams should return zero nutrition."""
    rice_id = CATEGORIES.index("rice")
    n = estimate_nutrition(class_id=rice_id, grams=0.0)
    assert n.calories == 0.0
    assert n.fat_g == 0.0


def test_to_dict_keys():
    """to_dict should produce the expected key set."""
    bread_id = CATEGORIES.index("bread")
    n = estimate_nutrition(class_id=bread_id, grams=150.0)
    d = n.to_dict()
    assert set(d.keys()) == {"food", "grams", "calories", "protein_g", "carbs_g", "fat_g"}


def test_all_categories_have_values():
    """All food categories should have nutrition values defined."""
    for i, category in enumerate(CATEGORIES):
        n = estimate_nutrition(class_id=i, grams=100.0)
        assert n.food == category
        assert n.grams == 100.0
        # All should have non-negative values
        assert n.calories >= 0
        assert n.protein_g >= 0
        assert n.carbs_g >= 0
        assert n.fat_g >= 0


def test_negative_class_id():
    """Negative class id should return unknown."""
    n = estimate_nutrition(class_id=-1, grams=100.0)
    assert n.food == "unknown"
    assert n.calories == 0