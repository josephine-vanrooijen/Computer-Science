## Data preprocessing

import json
import re

def normalize_text(text):
    if not isinstance(text, str):
        return text
    transformations = [
        (r'("|inch\b|\-inch)', ' inch'),
        (r'(hz|hertz|Hertz|HZ|\-hz)', ' hz'),
        (r'[^\w\s]', ''),  # Remove special characters
        (r'\s+', ' ')  # Replace multiple spaces with a single space
    ]
    text = text.lower()
    for pattern, replacement in transformations:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def map_brand_variations(brand):
    brand_mapping = {
        "lg electronics": "LG",
        "jvc tv": "JVC",
        "hello kitty": "Hello Kitty",
    }
    return brand_mapping.get(brand.lower(), brand)

def clean_data(file_path, output_path, brand_list):
    with open(file_path, 'r') as f:
        data = json.load(f)

    cleaned_data = {}
    brand_set = set(brand.lower() for brand in brand_list)  # Normalize brands to lowercase

    televisions_without_brand = []  # Collect televisions without a brand for debugging

    for model_id, products in data.items():
        cleaned_products = []
        for product in products:
            # Normalize title
            title = normalize_text(product.get('title', "")).lower()
            features_map = product.get('featuresMap', {})
            # Normalize featuresMap
            features_map = {k: normalize_text(v) for k, v in features_map.items()}

            # Determine brand from title
            title_words = title.split()
            brand = next((word for word in title_words if word in brand_set), None)

            # Fallback to normalized featuresMap['Brand']
            if not brand:
                fallback_brand = features_map.get('Brand', "")
                fallback_brand = map_brand_variations(fallback_brand)  # Map variations
                if fallback_brand.lower() in brand_set:
                    brand = fallback_brand

            # Mark as "Overig" if no brand is found
            if not brand:
                televisions_without_brand.append(product)  # Collect problematic entries
            features_map['Brand'] = brand or "Overig"

            cleaned_product = {
                "modelID": model_id,
                "title": title,
                "featuresMap": features_map,
                "brand": brand or "Overig"  # Include extracted brand directly
            }
            cleaned_products.append(cleaned_product)
        cleaned_data[model_id] = cleaned_products

    # Debugging: Print televisions without a brand
    print(f"Aantal televisies zonder merk: {len(televisions_without_brand)}")
    for product in televisions_without_brand:
        print(f"Product zonder brand: {product}")

    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

# Paths
input_path = "/Users/josephine/PycharmProjects/ComputerScienceTwo/pycharm-TVs-all-merged.json"
output_path = "/Users/josephine/PycharmProjects/ComputerScienceTwo/cleaned-data.json"

# Updated Brand List
brand_list = [
    "Apple", "Avue", "Coby", "Craig", "Curtisyoung", "Dynex", "Elite", "Haier", "Hannspree", "Hisense",
    "Insignia", "JVC", "LG", "Magnavox", "Mitsubishi", "Naxa", "NEC", "Panasonic", "Philips", "ProScan",
    "Pyle", "RCA", "Samsung", "Sansui", "Sanyo", "Sceptre", "Seiki", "Sharp", "Sony", "Supersonic", "TCL",
    "Toshiba", "Upstar", "Viewsonic", "Vizio", "Westinghouse", "SunBriteTV", "Optoma", "Venturer",
    "Compaq", "Azend", "Contex", "Affinity", "Epson", "Hiteker", "HP", "Elo", "Sigmac", "GPX", "Viore",
    "Hello Kitty"
]

# Execute the cleaning process
clean_data(input_path, output_path, brand_list)
