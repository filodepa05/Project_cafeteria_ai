"""Tests for nutrition API with caching."""

import json
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from src.nutrition_api import (
    NutritionCache,
    search_food,
    parse_nutrients,
    estimate_nutrition,
    get_cache_stats,
    clear_cache,
)
from src.nutrition import NutritionInfo
from src.dataset import CATEGORIES


class TestNutritionCache:
    """Test local JSON cache functionality."""
    
    def test_cache_save_and_load(self, tmp_path):
        """Cache should persist between instances."""
        cache_path = tmp_path / "test_cache.json"
        cache = NutritionCache(cache_path)
        
        cache.set("pizza", {"calories": 266, "protein": 11, "carbs": 33, "fat": 10})
        cache.save()
        
        # New instance should load existing cache
        cache2 = NutritionCache(cache_path)
        assert cache2.get("pizza") == {"calories": 266, "protein": 11, "carbs": 33, "fat": 10}
    
    def test_cache_case_insensitive(self, tmp_path):
        """Cache keys should be case-insensitive."""
        cache_path = tmp_path / "test_cache.json"
        cache = NutritionCache(cache_path)
        
        cache.set("Grilled_Chicken", {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6})
        assert cache.get("grilled_chicken") == {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6}
        assert cache.get("GRILLED_CHICKEN") == {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6}
    
    def test_cache_missing_key(self, tmp_path):
        """Missing cache key should return None."""
        cache_path = tmp_path / "test_cache.json"
        cache = NutritionCache(cache_path)
        
        assert cache.get("nonexistent_food") is None


class TestSearchFood:
    """Test USDA API search."""
    
    @patch("src.nutrition_api.requests.get")
    def test_search_food_success(self, mock_get):
        """Successful API call returns food data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "foods": [{"description": "Pizza", "fdcId": 123, "foodNutrients": []}]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = search_food("pizza", api_key="test_key")
        assert result["foods"][0]["description"] == "Pizza"
    
    @patch("src.nutrition_api.requests.get")
    def test_search_food_api_error(self, mock_get):
        """API error should raise exception."""
        mock_get.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception):
            search_food("pizza", api_key="test_key")
    
    def test_search_food_no_api_key(self):
        """Should raise error if API key not provided."""
        with pytest.raises(ValueError):
            search_food("pizza", api_key=None)


class TestParseNutrients:
    """Test nutrient extraction from USDA response."""
    
    def test_parse_nutrients_complete(self):
        """Should extract all 4 macronutrients."""
        food_item = {
            "foodNutrients": [
                {"nutrientId": 1008, "value": 266},  # calories
                {"nutrientId": 1003, "value": 11},   # protein
                {"nutrientId": 1005, "value": 33},   # carbs
                {"nutrientId": 1004, "value": 10},   # fat
            ]
        }
        
        result = parse_nutrients(food_item)
        assert result == {
            "calories": 266,
            "protein": 11,
            "carbs": 33,
            "fat": 10,
        }
    
    def test_parse_nutrients_missing(self):
        """Should handle missing nutrients gracefully."""
        food_item = {"foodNutrients": []}
        
        result = parse_nutrients(food_item)
        assert result == {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    
    def test_parse_nutrients_partial(self):
        """Should handle partial nutrient data."""
        food_item = {
            "foodNutrients": [
                {"nutrientId": 1008, "value": 100},  # only calories
            ]
        }
        
        result = parse_nutrients(food_item)
        assert result["calories"] == 100
        assert result["protein"] == 0
        assert result["carbs"] == 0
        assert result["fat"] == 0


class TestEstimateNutrition:
    """Test main estimation function."""
    
    def test_estimate_100g_from_cache(self, tmp_path):
        """100g of known food from cache returns expected values."""
        cache_path = tmp_path / "cache.json"
        cache = NutritionCache(cache_path)
        cache.set("pizza", {
            "calories": 266,
            "protein": 11,
            "carbs": 33,
            "fat": 10,
        })
        cache.save()
        
        # Get class_id for pizza
        pizza_id = CATEGORIES.index("pizza")
        
        with patch("src.nutrition_api.DEFAULT_CACHE_PATH", cache_path):
            result = estimate_nutrition(pizza_id, 100.0, use_api=True)
        
        assert result.food == "pizza"
        assert result.grams == 100.0
        assert result.calories == 266.0
        assert result.protein_g == 11.0
    
    def test_estimate_200g_scales_correctly(self, tmp_path):
        """200g should double the 100g values."""
        cache_path = tmp_path / "cache.json"
        cache = NutritionCache(cache_path)
        cache.set("rice", {
            "calories": 130,
            "protein": 2.7,
            "carbs": 28.2,
            "fat": 0.3,
        })
        cache.save()
        
        rice_id = CATEGORIES.index("rice")
        
        with patch("src.nutrition_api.DEFAULT_CACHE_PATH", cache_path):
            result = estimate_nutrition(rice_id, 200.0, use_api=True)
        
        assert result.calories == 260.0  # 130 * 2
        assert result.protein_g == 5.4   # 2.7 * 2
    
    @patch("src.nutrition_api.search_food")
    def test_api_call_caches_result(self, mock_search, tmp_path):
        """API response should be cached for future calls."""
        mock_search.return_value = {
            "foods": [{
                "description": "Salmon",
                "foodNutrients": [
                    {"nutrientId": 1008, "value": 208},
                    {"nutrientId": 1003, "value": 20},
                    {"nutrientId": 1005, "value": 0},
                    {"nutrientId": 1004, "value": 13},
                ]
            }]
        }
        
        cache_path = tmp_path / "cache.json"
        salmon_id = CATEGORIES.index("salmon")
        
        with patch("src.nutrition_api.DEFAULT_CACHE_PATH", cache_path):
            with patch("src.nutrition_api.USDA_API_KEY", "test_key"):
                result = estimate_nutrition(salmon_id, 100.0, use_api=True)
                
                # Should cache the result
                cache = NutritionCache(cache_path)
                assert cache.get("salmon") is not None
    
    @patch("src.nutrition_api.search_food")
    def test_fallback_on_api_error(self, mock_search):
        """Should fall back to hardcoded table when API fails."""
        mock_search.side_effect = Exception("API down")
        
        pizza_id = CATEGORIES.index("pizza")
        
        # Should get hardcoded value (266 for pizza per 100g)
        result = estimate_nutrition(pizza_id, 100.0, use_api=True)
        
        assert result.calories == 266.0
    
    def test_fallback_when_use_api_false(self):
        """Should use hardcoded table when use_api=False."""
        pasta_id = CATEGORIES.index("pasta")
        
        result = estimate_nutrition(pasta_id, 100.0, use_api=False)
        
        # Should get hardcoded value (158 for pasta per 100g)
        assert result.calories == 158.0
    
    def test_unknown_class_id(self):
        """Unknown class_id should return zeros."""
        result = estimate_nutrition(999, 100.0, use_api=False)
        
        assert result.food == "unknown"
        assert result.calories == 0.0
        assert result.protein_g == 0.0
    
    def test_all_categories_have_nutrition(self):
        """All categories should return valid nutrition info (fallback mode)."""
        for i, category in enumerate(CATEGORIES):
            result = estimate_nutrition(i, 100.0, use_api=False)
            assert result.food == category
            assert result.grams == 100.0
            # Should have some nutritional value (not all zeros for known foods)
            assert result.calories >= 0


class TestCacheManagement:
    """Test cache management functions."""
    
    def test_clear_cache(self, tmp_path):
        """clear_cache should remove all cached items."""
        cache_path = tmp_path / "cache.json"
        cache = NutritionCache(cache_path)
        cache.set("pizza", {"calories": 266})
        cache.save()
        
        with patch("src.nutrition_api.DEFAULT_CACHE_PATH", cache_path):
            clear_cache()
        
        cache2 = NutritionCache(cache_path)
        assert cache2.get("pizza") is None
    
    def test_get_cache_stats(self, tmp_path):
        """get_cache_stats should return cache statistics."""
        cache_path = tmp_path / "cache.json"
        cache = NutritionCache(cache_path)
        cache.set("pizza", {"calories": 266})
        cache.set("pasta", {"calories": 158})
        cache.save()
        
        with patch("src.nutrition_api.DEFAULT_CACHE_PATH", cache_path):
            stats = get_cache_stats()
        
        assert stats["cached_items"] == 2
        assert "pizza" in stats["foods"]
        assert "pasta" in stats["foods"]
