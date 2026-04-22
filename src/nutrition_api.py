"""nutrition_api.py – USDA FoodData Central API integration with local caching.

Replaces hardcoded nutrition table with live API calls.
Falls back to hardcoded table if API unavailable.

Usage:
    from src.nutrition_api import estimate_nutrition
    
    # Set API key via environment variable
    export USDA_API_KEY="your_api_key_here"
    
    # Or use the function directly
    nutrition = estimate_nutrition(0, 150.0)  # class_id=0 (pasta), 150g

The interface keeps class_id to maintain compatibility with existing code.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import requests

from src.nutrition import estimate_nutrition as fallback_estimate_nutrition
from src.nutrition import NutritionInfo
from src.dataset import CATEGORIES


# Configuration
USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
DEFAULT_CACHE_PATH = Path("data/nutrition_cache.json")


class NutritionCache:
    """Simple JSON file-based cache for API responses."""
    
    def __init__(self, cache_path: Path = DEFAULT_CACHE_PATH):
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        self._load()
    
    def _load(self):
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
    
    def save(self):
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)
    
    def get(self, food_name: str) -> Optional[dict]:
        """Get cached response for food_name."""
        return self._cache.get(food_name.lower())
    
    def set(self, food_name: str, data: dict):
        """Cache response for food_name."""
        self._cache[food_name.lower()] = data
        self.save()


def search_food(query: str, api_key: Optional[str] = None) -> dict:
    """Query USDA API for food items.
    
    Parameters
    ----------
    query : str
        Food name to search for (e.g., "pasta")
    api_key : str, optional
        USDA API key (defaults to USDA_API_KEY env var)
    
    Returns
    -------
    dict
        Raw USDA API response with food items and nutrients
    
    Raises
    ------
    requests.RequestException
        If API call fails
    ValueError
        If API key not provided
    """
    key = api_key or USDA_API_KEY
    if not key:
        raise ValueError("USDA_API_KEY not set. Get one at https://fdc.nal.usda.gov/api-key-signup.html")
    
    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "query": query,
        "api_key": key,
        "pageSize": 5,
        "dataType": "Foundation,SR%20Legacy",
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def parse_nutrients(food_item: dict) -> dict[str, float]:
    """Extract calories, protein, carbs, fat from USDA food item.
    
    USDA nutrient IDs:
    - 1008: Energy (kcal)
    - 1003: Protein (g)
    - 1005: Carbohydrate, by difference (g)
    - 1004: Total lipid (fat) (g)
    
    Parameters
    ----------
    food_item : dict
        Food item from USDA API response
    
    Returns
    -------
    dict[str, float]
        Dict with keys: calories, protein, carbs, fat (all per 100g)
    """
    nutrients = {
        "calories": 0.0,
        "protein": 0.0,
        "carbs": 0.0,
        "fat": 0.0,
    }
    
    nutrient_map = {
        1008: "calories",
        1003: "protein",
        1005: "carbs",
        1004: "fat",
    }
    
    for nutrient in food_item.get("foodNutrients", []):
        nutrient_id = nutrient.get("nutrientId")
        if nutrient_id in nutrient_map:
            key = nutrient_map[nutrient_id]
            nutrients[key] = nutrient.get("value", 0.0)
    
    return nutrients


def estimate_nutrition(class_id: int, grams: float, use_api: bool = True) -> NutritionInfo:
    """Estimate nutrition for a food item using USDA API with caching.
    
    This function maintains the same interface as src.nutrition.estimate_nutrition
    (using class_id instead of food name) for backward compatibility.
    
    Parameters
    ----------
    class_id : int
        Index into CATEGORIES (0-42)
    grams : float
        Portion weight in grams
    use_api : bool
        Whether to try using the API (falls back to hardcoded if False or API fails)
    
    Returns
    -------
    NutritionInfo
        Scaled nutrition information
    
    Notes
    -----
    1. Check local cache first
    2. If not cached, call USDA API
    3. Parse response for calories, protein, carbs, fat
    4. Scale by (grams / 100)
    5. Save to cache
    6. Return NutritionInfo
    7. If API fails or use_api=False, fall back to hardcoded table
    """
    # Validate class_id
    if class_id < 0 or class_id >= len(CATEGORIES):
        return NutritionInfo("unknown", grams, 0.0, 0.0, 0.0, 0.0)
    
    food_name = CATEGORIES[class_id]
    
    # If not using API, fall back immediately
    if not use_api:
        return fallback_estimate_nutrition(class_id, grams)
    
    cache = NutritionCache()
    
    # 1. Check cache
    cached = cache.get(food_name)
    if cached:
        per100 = cached
    else:
        try:
            # 2. Query API
            response = search_food(food_name)
            
            if not response.get("foods"):
                # No results - fall back
                return fallback_estimate_nutrition(class_id, grams)
            
            # 3. Use first result
            food_item = response["foods"][0]
            per100 = parse_nutrients(food_item)
            
            # 4. Save to cache
            cache.set(food_name, per100)
            
        except (requests.RequestException, ValueError) as e:
            # API unavailable or error - fall back
            print(f"API error for {food_name}: {e}. Using fallback.")
            return fallback_estimate_nutrition(class_id, grams)
    
    # 5. Scale by grams
    factor = grams / 100.0
    
    return NutritionInfo(
        food=food_name,
        grams=round(grams, 1),
        calories=round(per100["calories"] * factor, 1),
        protein_g=round(per100["protein"] * factor, 1),
        carbs_g=round(per100["carbs"] * factor, 1),
        fat_g=round(per100["fat"] * factor, 1),
    )


def clear_cache():
    """Clear the nutrition cache."""
    cache = NutritionCache()
    cache._cache = {}
    cache.save()
    print(f"Cache cleared: {cache.cache_path}")


def get_cache_stats() -> dict:
    """Get statistics about the nutrition cache.
    
    Returns
    -------
    dict
        Statistics about cached items
    """
    cache = NutritionCache()
    return {
        "cache_path": str(cache.cache_path),
        "cached_items": len(cache._cache),
        "foods": list(cache._cache.keys()),
    }
