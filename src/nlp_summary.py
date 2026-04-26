VEGETABLES_AND_FRUITS = {
    "apple", "banana", "orange", "tomato", "lettuce", "spinach", "kale",
    "broccoli", "carrot", "cucumber", "pepper", "salad", "greens", "vegetable", "fruit"
}

def _get_highest_macro(totals):
    macros = {
        "protein": totals.get("protein_g", 0),
        "carbohydrates": totals.get("carbs_g", 0),
        "healthy fats": totals.get("fat_g", 0),
    }
    return max(macros, key=macros.get)

def _has_vegetables_or_fruits(items):
    for item in items:
        food_name = item.get("food", "").lower()
        for keyword in VEGETABLES_AND_FRUITS:
            if keyword in food_name:
                return True
    return False

def generate_summary(analysis):
    totals = analysis.get("totals", {})
    items = analysis.get("items", [])
    calories = totals.get("calories", 0)
    protein = totals.get("protein_g", 0)
    fat = totals.get("fat_g", 0)
    sentences = []
    if calories > 800:
        sentences.append("This is a high-calorie meal.")
    elif calories < 400:
        sentences.append("This is a light meal. Consider adding more if this is your main meal.")
    else:
        sentences.append(f"This meal provides {int(calories)} kcal, a moderate amount.")
    if protein < 15:
        sentences.append("Protein is low. Consider adding chicken, eggs, or legumes.")
    fat_calories = fat * 9
    if calories > 0 and (fat_calories / calories) > 0.40:
        sentences.append("This meal is high in fat.")
    if not _has_vegetables_or_fruits(items):
        sentences.append("Adding vegetables would improve nutritional balance.")
    highest_macro = _get_highest_macro(totals)
    sentences.append(f"Good source of {highest_macro}.")
    return " ".join(sentences)
