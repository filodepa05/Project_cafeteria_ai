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
    "apple_pie": None,
    "baby_back_ribs": "pork_ribs",
    "baklava": "cake_pastry",
    "beef_carpaccio": "grilled_beef",
    "beef_tartare": "grilled_beef",
    "beet_salad": "salad",
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
    "cheese_plate": None,
    "chicken_curry": "chicken_stew",
    "chicken_quesadilla": "wrap_sandwich",
    "chicken_wings": "fried_chicken",
    "chocolate_cake": "cake_pastry",
    "chocolate_mousse": "cake_pastry",
    "churros": "cake_pastry",
    "clam_chowder": "soup_cream",
    "club_sandwich": "wrap_sandwich",
    "crab_cakes": "fried_fish",
    "creme_brulee": "cake_pastry",
    "croque_madame": None,
    "cup_cakes": "cake_pastry",
    "deviled_eggs": "eggs",
    "donuts": "cake_pastry",
    "dumplings": None,
    "edamame": None,
    "eggs_benedict": "eggs",
    "escargots": None,
    "falafel": None,
    "filet_mignon": "grilled_beef",
    "fish_and_chips": "fried_fish",
    "foie_gras": None,
    "french_fries": "fries",
    "french_onion_soup": "soup_cream",
    "french_toast": "bread",
    "fried_calamari": "fried_fish",
    "fried_rice": "rice",
    "frozen_yogurt": "yogurt",
    "garlic_bread": "bread",
    "gnocchi": "pasta",
    "greek_salad": "salad",
    "grilled_cheese_sandwich": "wrap_sandwich",
    "grilled_salmon": "salmon",
    "guacamole": None,
    "gyoza": None,
    "hamburger": "grilled_beef",
    "hot_and_sour_soup": "soup_cream",
    "hot_dog": None,
    "huevos_rancheros": "eggs",
    "hummus": None,
    "ice_cream": "ice_cream_sorbet",
    "lasagna": "lasagne",
    "lobster_bisque": "soup_cream",
    "lobster_roll_sandwich": "wrap_sandwich",
    "macaroni_and_cheese": "pasta",
    "macarons": "cake_pastry",
    "miso_soup": "soup_cream",
    "mussels": None,
    "nachos": "fries",
    "omelette": "eggs",
    "onion_rings": "fried_potatoes",
    "oysters": None,
    "pad_thai": "pasta",
    "paella": "rice",
    "pancakes": "cake_pastry",
    "panna_cotta": "cake_pastry",
    "peking_duck": None,
    "pho": "soup_cream",
    "pizza": "pizza",
    "pork_chop": "grilled_pork",
    "poutine": "fries",
    "prime_rib": "grilled_beef",
    "pulled_pork_sandwich": "grilled_pork",
    "ramen": "soup_cream",
    "ravioli": "pasta",
    "red_velvet_cake": "cake_pastry",
    "risotto": "rice",
    "samosa": None,
    "sashimi": "grilled_fish",
    "scallops": "grilled_fish",
    "seaweed_salad": "salad",
    "shrimp_and_grits": None,
    "spaghetti_bolognese": "pasta",
    "spaghetti_carbonara": "pasta",
    "spring_rolls": None,
    "steak": "grilled_beef",
    "strawberry_shortcake": "cake_pastry",
    "sushi": "rice",
    "tacos": "wrap_sandwich",
    "takoyaki": None,
    "tiramisu": "cake_pastry",
    "tuna_tartare": "tuna",
    "waffles": "cake_pastry",
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
