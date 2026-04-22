"""class_mapping.py – Mapping from source dataset classes to Filo's master categories."""

from __future__ import annotations

# Filo's master categories (43 classes)
FILO_CATEGORIES = [
    "pasta", "rice", "pizza", "bread", "fries", "couscous", "potatoes", "wrap_sandwich",
    "grilled_chicken", "fried_chicken", "chicken_stew", "turkey",
    "grilled_beef", "beef_stew", "meatballs", "grilled_pork", "pork_ribs",
    "salmon", "hake", "tuna", "cod", "grilled_fish", "fried_fish",
    "eggs", "lentils", "chickpeas",
    "salad", "soup_cream", "grilled_vegetables", "sauteed_vegetables", "broccoli", "stuffed_peppers",
    "poke_bowl", "lasagne",
    "fresh_fruit", "fruit_salad",
    "yogurt", "cake_pastry", "ice_cream_sorbet", "juice_drink",
    "rotisserie_chicken", "fried_potatoes", "baked_potatoes",
]

# Create reverse lookup
FILO_TO_IDX = {name: idx for idx, name in enumerate(FILO_CATEGORIES)}

# UNIMIB2016 class mapping to Filo categories
# Source: UNIMIB2016 has 73 food classes with polygon annotations
UNIMIB_TO_FILO = {
    # Starches & grains
    "pasta_pesto": "pasta",
    "pasta_tomato_sauce": "pasta",
    "pasta_meat_sauce": "pasta",
    "pasta_clams": "pasta",
    "pasta_squid": "pasta",
    "risotto": "rice",
    "rice_salad": "rice",
    "pizza": "pizza",
    "pizza_margherita": "pizza",
    "bread": "bread",
    "focaccia": "bread",
    "sandwich": "wrap_sandwich",
    "sandwich_chicken": "wrap_sandwich",
    "sandwich_ham": "wrap_sandwich",
    "french_fries": "fries",
    "potatoes": "potatoes",
    "mashed_potatoes": "potatoes",
    
    # Poultry
    "grilled_chicken_breast": "grilled_chicken",
    "chicken_nuggets": "fried_chicken",
    "chicken_wings": "fried_chicken",
    "chicken_curry": "chicken_stew",
    "chicken_thighs": "chicken_stew",
    "turkey": "turkey",
    
    # Meat
    "beef_steak": "grilled_beef",
    "beef_carpaccio": "grilled_beef",
    "beef_tartare": "grilled_beef",
    "meatballs": "meatballs",
    "pork_cutlet": "grilled_pork",
    "pork_loin": "grilled_pork",
    
    # Fish
    "salmon": "salmon",
    "grilled_salmon": "salmon",
    "hake": "hake",
    "grilled_hake": "hake",
    "tuna": "tuna",
    "grilled_tuna": "tuna",
    "cod": "cod",
    "fried_cod": "fried_fish",
    "fish": "grilled_fish",
    "fried_fish": "fried_fish",
    "fish_sticks": "fried_fish",
    
    # Eggs
    "omelette": "eggs",
    "scrambled_eggs": "eggs",
    "fried_eggs": "eggs",
    
    # Legumes
    "lentils": "lentils",
    "beans": "lentils",
    "chickpeas": "chickpeas",
    
    # Vegetables
    "salad": "salad",
    "caesar_salad": "salad",
    "mixed_salad": "salad",
    "soup": "soup_cream",
    "cream_soup": "soup_cream",
    "vegetables": "grilled_vegetables",
    "grilled_vegetables": "grilled_vegetables",
    "sauteed_vegetables": "sauteed_vegetables",
    "broccoli": "broccoli",
    "stuffed_peppers": "stuffed_peppers",
    
    # Fruit & desserts
    "fruit": "fresh_fruit",
    "fruit_salad": "fruit_salad",
    "yogurt": "yogurt",
    "cake": "cake_pastry",
    "tart": "cake_pastry",
    "muffin": "cake_pastry",
    "croissant": "cake_pastry",
    "ice_cream": "ice_cream_sorbet",
    
    # Drinks
    "juice": "juice_drink",
    "smoothie": "juice_drink",
    
    # Other
    "lasagne": "lasagne",
    "poke_bowl": "poke_bowl",
    "roasted_chicken": "rotisserie_chicken",
    "baked_potatoes": "baked_potatoes",
}

# Food-101 class mapping to Filo categories
# Source: Food-101 has 101 classes, we map only relevant ones
FOOD101_TO_FILO = {
    # Starches & grains (0-7)
    "apple_pie": "cake_pastry",  # CHANGED from None - it's a dessert
    "baby_back_ribs": "pork_ribs",
    "baklava": "cake_pastry",
    "beef_carpaccio": "grilled_beef",
    "beef_tartare": "grilled_beef",
    "beet_salad": "salad",  # CHANGED from None
    "beignets": "cake_pastry",
    "bibimbap": "rice",
    "bread_pudding": "cake_pastry",
    "breakfast_burrito": "wrap_sandwich",
    "bruschetta": "bread",
    "caesar_salad": "salad",
    "cannoli": "cake_pastry",
    "caprese_salad": "salad",
    "carrot_cake": "cake_pastry",
    "ceviche": "grilled_fish",
    "cheesecake": "cake_pastry",
    "cheese_plate": "cake_pastry",  # CHANGED from None - often served as dessert course
    "chicken_curry": "chicken_stew",
    "chicken_quesadilla": "wrap_sandwich",
    "chicken_wings": "fried_chicken",
    "chocolate_cake": "cake_pastry",
    "chocolate_mousse": "cake_pastry",
    "churros": "cake_pastry",
    "clam_chowder": "soup_cream",  # CHANGED from None
    "club_sandwich": "wrap_sandwich",
    "crab_cakes": "fried_fish",
    "creme_brulee": "cake_pastry",
    "croque_madame": "wrap_sandwich",  # CHANGED from None - French grilled sandwich
    "cup_cakes": "cake_pastry",
    "deviled_eggs": "eggs",  # CHANGED from None
    "donuts": "cake_pastry",
    "dumplings": "wrap_sandwich",  # CHANGED from None - filled wraps
    "edamame": "sauteed_vegetables",  # CHANGED from None
    "eggs_benedict": "eggs",  # CHANGED from None
    "escargots": None,  # Keep as None - snails unlikely in cafeteria
    "falafel": "chickpeas",  # CHANGED from None - chickpea fritters
    "filet_mignon": "grilled_beef",
    "fish_and_chips": "fried_fish",
    "foie_gras": None,  # Keep as None - specialty item
    "french_fries": "fries",
    "french_onion_soup": "soup_cream",  # CHANGED from None
    "french_toast": "bread",  # CHANGED from None - breakfast item
    "fried_calamari": "fried_fish",
    "fried_rice": "rice",
    "frozen_yogurt": "yogurt",
    "garlic_bread": "bread",
    "gnocchi": "pasta",
    "greek_salad": "salad",
    "grilled_cheese_sandwich": "wrap_sandwich",
    "grilled_salmon": "salmon",
    "guacamole": "sauteed_vegetables",  # CHANGED from None - avocado dip, map to vegetables
    "gyoza": "wrap_sandwich",  # CHANGED from None - dumplings
    "hamburger": "grilled_beef",
    "hot_and_sour_soup": "soup_cream",  # CHANGED from None
    "hot_dog": "wrap_sandwich",  # CHANGED from None - bread + meat
    "huevos_rancheros": "eggs",  # CHANGED from None
    "hummus": "chickpeas",  # CHANGED from None
    "ice_cream": "ice_cream_sorbet",
    "lasagna": "lasagne",
    "lobster_bisque": "soup_cream",  # CHANGED from None
    "lobster_roll_sandwich": "wrap_sandwich",
    "macaroni_and_cheese": "pasta",
    "macarons": "cake_pastry",
    "miso_soup": "soup_cream",  # CHANGED from None
    "mussels": "fried_fish",  # CHANGED from None - seafood, closest match
    "nachos": "fries",  # CHANGED from None - similar to chips
    "omelette": "eggs",
    "onion_rings": "fried_potatoes",  # CHANGED from None
    "oysters": None,  # Keep as None - specialty seafood
    "pad_thai": "pasta",
    "paella": "rice",
    "pancakes": "cake_pastry",  # CHANGED from None - breakfast sweet
    "panna_cotta": "cake_pastry",
    "peking_duck": None,  # Keep as None - specialty Chinese dish
    "pho": "soup_cream",  # CHANGED from None - Vietnamese noodle soup
    "pizza": "pizza",
    "pork_chop": "grilled_pork",
    "poutine": "fried_potatoes",  # CHANGED from None - fries with gravy
    "prime_rib": "grilled_beef",
    "pulled_pork_sandwich": "grilled_pork",
    "ramen": "soup_cream",  # CHANGED from None - Japanese noodle soup
    "ravioli": "pasta",
    "red_velvet_cake": "cake_pastry",
    "risotto": "rice",
    "samosa": "wrap_sandwich",  # CHANGED from None - filled pastry
    "sashimi": "grilled_fish",
    "scallops": "grilled_fish",
    "seaweed_salad": "salad",  # CHANGED from None
    "shrimp_and_grits": "rice",  # CHANGED from None - closest match
    "spaghetti_bolognese": "pasta",
    "spaghetti_carbonara": "pasta",
    "spring_rolls": "wrap_sandwich",  # CHANGED from None
    "steak": "grilled_beef",
    "strawberry_shortcake": "cake_pastry",
    "sushi": "rice",
    "tacos": "wrap_sandwich",
    "takoyaki": None,  # Keep as None - specialty Japanese octopus balls
    "tiramisu": "cake_pastry",
    "tuna_tartare": "tuna",
    "waffles": "cake_pastry",  # CHANGED from None
}

def get_filo_class_id(source_class: str, mapping: dict[str, str | None]) -> int | None:
    """Map a source class name to Filo class ID.
    
    Parameters
    ----------
    source_class : str
        Class name from source dataset
    mapping : dict
        Mapping dict (e.g., UNIMIB_TO_FILO or FOOD101_TO_FILO)
    
    Returns
    -------
    int or None
        Filo class ID, or None if not mapped
    """
    filo_name = mapping.get(source_class.lower())
    if filo_name is None:
        return None
    return FILO_TO_IDX.get(filo_name)


def get_filo_class_name(source_class: str, mapping: dict[str, str | None]) -> str | None:
    """Map a source class name to Filo class name.
    
    Parameters
    ----------
    source_class : str
        Class name from source dataset
    mapping : dict
        Mapping dict (e.g., UNIMIB_TO_FILO or FOOD101_TO_FILO)
    
    Returns
    -------
    str or None
        Filo class name, or None if not mapped
    """
    return mapping.get(source_class.lower())
