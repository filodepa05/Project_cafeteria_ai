"""
nutrition.py – Maps a food label + estimated grams → calories & macros.

All data comes from config.NUTRITION_PER_100G (USDA approximations).
"""

from __future__ import annotations

from dataclasses import dataclass

from config import NUTRITION_PER_100G


@dataclass
class NutritionEstimate:
    """Nutritional breakdown for one detected item."""
    food: str
    grams: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def to_dict(self) -> dict:
        return {
            "food": self.food,
            "estimated_grams": self.grams,
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
        }


def lookup(food: str, grams: float) -> NutritionEstimate:
    """Return a NutritionEstimate for `grams` of `food`."""

    per100 = NUTRITION_PER_100G.get(food)
    if per100 is None:
        # Unknown food – return zeroes so the pipeline doesn't crash.
        return NutritionEstimate(food, grams, 0, 0, 0, 0)

    factor = grams / 100.0
    return NutritionEstimate(
        food=food,
        grams=round(grams, 1),
        calories=round(per100["calories"] * factor, 1),
        protein_g=round(per100["protein"] * factor, 1),
        carbs_g=round(per100["carbs"] * factor, 1),
        fat_g=round(per100["fat"] * factor, 1),
    )