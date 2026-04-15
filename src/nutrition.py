"""
nutrition.py – Food → calorie / macronutrient lookup.

Uses approximate USDA values per 100 g.  A later milestone
could swap this for an API call or a learned model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from src.dataset import CATEGORIES


# ── Per-100g values (approx. USDA) ────────────────────────────────
# Indexed in the same order as CATEGORIES so we can look up by class id.
_DB: list[dict[str, float]] = [
    # banana
    {"calories": 89,  "protein": 1.1, "carbs": 22.8, "fat": 0.3},
    # apple
    {"calories": 52,  "protein": 0.3, "carbs": 13.8, "fat": 0.2},
    # sandwich
    {"calories": 250, "protein": 11.0, "carbs": 28.0, "fat": 10.0},
    # orange
    {"calories": 47,  "protein": 0.9, "carbs": 11.8, "fat": 0.1},
    # broccoli
    {"calories": 34,  "protein": 2.8, "carbs": 6.6,  "fat": 0.4},
    # carrot
    {"calories": 41,  "protein": 0.9, "carbs": 9.6,  "fat": 0.2},
    # hot_dog
    {"calories": 290, "protein": 10.0, "carbs": 24.0, "fat": 18.0},
    # pizza
    {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    # donut
    {"calories": 452, "protein": 5.0, "carbs": 51.0, "fat": 25.0},
    # cake
    {"calories": 350, "protein": 4.5, "carbs": 50.0, "fat": 15.0},
]


@dataclass
class NutritionInfo:
    food: str
    grams: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_nutrition(class_id: int, grams: float) -> NutritionInfo:
    """Look up nutrition for a given class id and portion weight.

    Parameters
    ----------
    class_id : int   – index into CATEGORIES
    grams    : float – estimated portion weight

    Returns
    -------
    NutritionInfo with scaled macros.
    """
    if class_id < 0 or class_id >= len(_DB):
        return NutritionInfo("unknown", grams, 0, 0, 0, 0)

    per100 = _DB[class_id]
    factor = grams / 100.0
    return NutritionInfo(
        food=CATEGORIES[class_id],
        grams=round(grams, 1),
        calories=round(per100["calories"] * factor, 1),
        protein_g=round(per100["protein"] * factor, 1),
        carbs_g=round(per100["carbs"] * factor, 1),
        fat_g=round(per100["fat"] * factor, 1),
    )