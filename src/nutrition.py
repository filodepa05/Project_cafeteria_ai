"""
nutrition.py – Food → calorie / macronutrient lookup.

Uses approximate USDA / standard reference values per 100 g.
Indexed in the same order as CATEGORIES in dataset.py.

A later milestone swaps this for live USDA FoodData Central API
calls with local caching (see src/nutrition_api.py — JP's task).

Class list (40 items) derived from:
  • IE Tower weekly rotating menu (Oct 2024)
  • Do Eat! cafeteria menu (María de Molina)
  • Campus Segovia Eurest menu (Autumn 2023)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from src.dataset import CATEGORIES


# ── Per-100g reference values ──────────────────────────────────────
# Format: {"calories": kcal, "protein": g, "carbs": g, "fat": g}
# Sources: USDA FoodData Central, standard recipe references.
# Values are for the prepared/cooked item as it appears on the tray.
# Indexed to match CATEGORIES exactly — DO NOT reorder.

_DB: list[dict[str, float]] = [
    # 0  pasta (cooked, plain / with sauce — mid estimate)
    {"calories": 158, "protein": 5.8,  "carbs": 30.9, "fat": 0.9},
    # 1  rice (cooked, white / basmati / paella estimate)
    {"calories": 130, "protein": 2.7,  "carbs": 28.2, "fat": 0.3},
    # 2  pizza (cheese/tomato, average slice)
    {"calories": 266, "protein": 11.0, "carbs": 33.0, "fat": 10.0},
    # 3  bread (white / baguette)
    {"calories": 265, "protein": 9.0,  "carbs": 49.0, "fat": 3.2},
    # 4  fries (oven / fried)
    {"calories": 312, "protein": 3.4,  "carbs": 41.4, "fat": 15.0},
    # 5  couscous (cooked)
    {"calories": 112, "protein": 3.8,  "carbs": 23.2, "fat": 0.2},
    # 6  potatoes (boiled/gratin average)
    {"calories": 87,  "protein": 1.9,  "carbs": 20.1, "fat": 0.1},
    # 7  wrap_sandwich (flour tortilla + typical filling)
    {"calories": 220, "protein": 10.0, "carbs": 28.0, "fat": 7.0},
    # 8  grilled_chicken (breast, no skin)
    {"calories": 165, "protein": 31.0, "carbs": 0.0,  "fat": 3.6},
    # 9  fried_chicken (breaded)
    {"calories": 246, "protein": 22.5, "carbs": 11.0, "fat": 12.0},
    # 10 chicken_stew (with sauce / arroz con pollo estimate)
    {"calories": 175, "protein": 22.0, "carbs": 5.0,  "fat": 7.0},
    # 11 turkey (grilled/roasted cutlet)
    {"calories": 149, "protein": 29.0, "carbs": 0.0,  "fat": 3.0},
    # 12 grilled_beef (entrecot / filete plancha)
    {"calories": 217, "protein": 26.0, "carbs": 0.0,  "fat": 12.0},
    # 13 beef_stew (ragout / goulash / chili con carne)
    {"calories": 190, "protein": 18.0, "carbs": 8.0,  "fat": 9.0},
    # 14 meatballs (in sauce)
    {"calories": 195, "protein": 13.5, "carbs": 8.0,  "fat": 12.0},
    # 15 grilled_pork (secreto / lomo / solomillo)
    {"calories": 212, "protein": 27.0, "carbs": 0.0,  "fat": 11.0},
    # 16 pork_ribs (baked / caramelised)
    {"calories": 292, "protein": 19.0, "carbs": 4.0,  "fat": 22.0},
    # 17 salmon (grilled / baked)
    {"calories": 208, "protein": 20.0, "carbs": 0.0,  "fat": 13.0},
    # 18 hake (merluza plancha / crujiente)
    {"calories": 90,  "protein": 18.0, "carbs": 0.0,  "fat": 1.3},
    # 19 tuna (emperador / atún plancha)
    {"calories": 144, "protein": 23.3, "carbs": 0.0,  "fat": 4.9},
    # 20 cod (bacalao)
    {"calories": 82,  "protein": 17.8, "carbs": 0.0,  "fat": 0.7},
    # 21 grilled_fish (generic white fish: perca, lubina, rape)
    {"calories": 96,  "protein": 20.0, "carbs": 0.0,  "fat": 1.8},
    # 22 fried_fish (breaded / rebozado / calamares)
    {"calories": 228, "protein": 14.0, "carbs": 18.0, "fat": 11.0},
    # 23 eggs (fried / scrambled / omelette — 2 eggs ≈ 100g)
    {"calories": 155, "protein": 11.0, "carbs": 1.1,  "fat": 11.0},
    # 24 lentils (cooked / stewed)
    {"calories": 116, "protein": 9.0,  "carbs": 20.1, "fat": 0.4},
    # 25 chickpeas (cooked / curry)
    {"calories": 164, "protein": 8.9,  "carbs": 27.4, "fat": 2.6},
    # 26 salad (mixed green / caesar with dressing)
    {"calories": 65,  "protein": 3.0,  "carbs": 6.0,  "fat": 3.5},
    # 27 soup_cream (vegetable cream / pumpkin / mushroom — average)
    {"calories": 72,  "protein": 2.0,  "carbs": 10.0, "fat": 3.0},
    # 28 grilled_vegetables (peppers, zucchini, tomatoes, ratatouille)
    {"calories": 45,  "protein": 1.8,  "carbs": 8.0,  "fat": 1.5},
    # 29 sauteed_vegetables (green beans, peas, mushrooms, setas)
    {"calories": 60,  "protein": 2.5,  "carbs": 9.0,  "fat": 2.0},
    # 30 broccoli (gratin / steamed / chowder)
    {"calories": 55,  "protein": 3.7,  "carbs": 7.0,  "fat": 1.5},
    # 31 stuffed_peppers (with tuna or vegetarian filling)
    {"calories": 115, "protein": 7.0,  "carbs": 10.0, "fat": 5.0},
    # 32 poke_bowl (tuna or salmon, rice, vegetables — per 100g of bowl)
    {"calories": 145, "protein": 10.0, "carbs": 17.0, "fat": 4.0},
    # 33 lasagne (meat or vegetable)
    {"calories": 135, "protein": 8.0,  "carbs": 13.0, "fat": 5.5},
    # 34 fresh_fruit (apple / orange / melon — average)
    {"calories": 52,  "protein": 0.5,  "carbs": 13.0, "fat": 0.2},
    # 35 fruit_salad (macedonia — mixed fresh fruit)
    {"calories": 55,  "protein": 0.8,  "carbs": 14.0, "fat": 0.2},
    # 36 yogurt (plain / natillas)
    {"calories": 61,  "protein": 3.5,  "carbs": 4.7,  "fat": 3.3},
    # 37 cake_pastry (tarta / brownie / croissant / cookie — average)
    {"calories": 380, "protein": 5.5,  "carbs": 50.0, "fat": 18.0},
    # 38 ice_cream_sorbet (batido / sorbete — average)
    {"calories": 195, "protein": 3.5,  "carbs": 28.0, "fat": 8.0},
    # 39 juice_drink (zumo / smoothie — average)
    {"calories": 45,  "protein": 0.5,  "carbs": 11.0, "fat": 0.1},
    # 40 rotisserie_chicken (whole roasted, skin on — per 100g edible portion)
    {"calories": 190, "protein": 27.0, "carbs": 0.0,  "fat": 9.0},
    # 41 fried_potatoes (patatas bravas / wedges / thick-cut fried)
    {"calories": 265, "protein": 3.0,  "carbs": 35.0, "fat": 13.0},
    # 42 baked_potatoes (patatas al horno / gratinadas / panaderas)
    {"calories": 150, "protein": 3.5,  "carbs": 30.0, "fat": 2.5},
]

# Sanity check — catches mistakes if someone edits one list but not the other
assert len(_DB) == len(CATEGORIES), (
    f"_DB has {len(_DB)} entries but CATEGORIES has {len(CATEGORIES)}. "
    "Keep them in sync!"
)


# ── Data class ────────────────────────────────────────────────────

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


# ── Lookup function ───────────────────────────────────────────────

def estimate_nutrition(class_id: int, grams: float) -> NutritionInfo:
    """Return scaled macronutrients for a given food class and portion.

    Parameters
    ----------
    class_id : int   – index into CATEGORIES (0–39)
    grams    : float – estimated portion weight in grams

    Returns
    -------
    NutritionInfo with macros scaled to the given portion weight.
    Falls back to an all-zero 'unknown' record for out-of-range ids.
    """
    if class_id < 0 or class_id >= len(_DB):
        return NutritionInfo("unknown", grams, 0.0, 0.0, 0.0, 0.0)

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