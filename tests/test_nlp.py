import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nlp_summary import generate_summary

def test_high_calorie():
    result = generate_summary({
        "items": [{"food": "burger", "grams": 300, "calories": 900}],
        "totals": {"calories": 900, "protein_g": 40, "carbs_g": 60, "fat_g": 30}
    })
    assert "high-calorie" in result, f"FAIL: {result}"
    print("✅ Rule 1 - High calorie OK")

def test_light_meal():
    result = generate_summary({
        "items": [{"food": "salad", "grams": 100, "calories": 200}],
        "totals": {"calories": 200, "protein_g": 20, "carbs_g": 10, "fat_g": 5}
    })
    assert "light meal" in result, f"FAIL: {result}"
    print("✅ Rule 2 - Light meal OK")

def test_low_protein():
    result = generate_summary({
        "items": [{"food": "bread", "grams": 100, "calories": 500}],
        "totals": {"calories": 500, "protein_g": 5, "carbs_g": 80, "fat_g": 10}
    })
    assert "Protein is low" in result, f"FAIL: {result}"
    print("✅ Rule 3 - Low protein OK")

def test_high_fat():
    result = generate_summary({
        "items": [{"food": "cheese", "grams": 100, "calories": 600}],
        "totals": {"calories": 600, "protein_g": 20, "carbs_g": 10, "fat_g": 40}
    })
    assert "high in fat" in result, f"FAIL: {result}"
    print("✅ Rule 4 - High fat OK")

def test_no_vegetables():
    result = generate_summary({
        "items": [{"food": "pasta", "grams": 200, "calories": 600}],
        "totals": {"calories": 600, "protein_g": 20, "carbs_g": 80, "fat_g": 10}
    })
    assert "vegetables" in result, f"FAIL: {result}"
    print("✅ Rule 5 - No vegetables OK")

def test_positive_ending():
    result = generate_summary({
        "items": [{"food": "chicken", "grams": 200, "calories": 500}],
        "totals": {"calories": 500, "protein_g": 50, "carbs_g": 10, "fat_g": 10}
    })
    assert "Good source of" in result, f"FAIL: {result}"
    print("✅ Rule 6 - Positive ending OK")

if __name__ == "__main__":
    test_high_calorie()
    test_light_meal()
    test_low_protein()
    test_high_fat()
    test_no_vegetables()
    test_positive_ending()
    print("\n🎉 All tests passed!")
