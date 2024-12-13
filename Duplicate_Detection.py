# Import statements
import json
import random
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Definities van functies zoals minhash_signature
def minhash_signature(one_hot, hash_functions, max_val):
    signature = []
    for a, b in hash_functions:
        min_hash = float('inf')
        for idx, value in enumerate(one_hot):
            if value == 1:
                hash_val = (a * idx + b) % max_val
                min_hash = min(min_hash, hash_val)
        signature.append(min_hash)
    return signature


def bootstrap_data(data, seed=None):
    if seed is not None:
        random.seed(seed)

    total_indices = list(range(len(data)))
    train_indices = random.choices(total_indices, k=int(len(data) * 0.63))
    test_indices = list(set(total_indices) - set(train_indices))

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data


def evaluate_with_bootstrapping(products, num_bootstraps=5, threshold=0.3):
    results = []
    for i in range(num_bootstraps):
        print(f"Running bootstrap {i + 1}/{num_bootstraps}...")

        # Bootstrapping
        train_data, test_data = bootstrap_data(products, seed=i)
        print(f"Aantal producten in train_data: {len(train_data)}")
        print(f"Aantal producten in test_data: {len(test_data)}")

        # Preprocessing train_data
        for product in train_data:
            product['shingles'] = generate_shingles(product['title'], features=product['featuresMap'])
            product['model_words'] = extract_model_words(product['title'])
        generate_one_hot_vectors(train_data, global_shingles_vocab)

        for product in train_data:
            product['minhash_signature'] = minhash_signature(product['one_hot'], hash_functions, max_val)

        # Clustering train_data
        candidate_pairs = banding_method(train_data, num_bands=10, rows_per_band=10)
        print(f"Aantal kandidaatparen: {len(candidate_pairs)}")
        clusters = hierarchical_clustering(train_data, candidate_pairs, threshold=threshold)
        print(f"Aantal clusters: {len(set(clusters))}")

        # Preprocessing test_data
        for product in test_data:
            product['shingles'] = generate_shingles(product['title'], features=product['featuresMap'])
            product['model_words'] = extract_model_words(product['title'])
            generate_one_hot_vectors([product], global_shingles_vocab)

        # Evaluate clusters
        result = evaluate_clusters(test_data, clusters, train_data)
        results.append(result)

    avg_results = {
        "Pair Quality": np.mean([res["Pair Quality"] for res in results]),
        "Pair Completeness": np.mean([res["Pair Completeness"] for res in results]),
        "F1*-Measure": np.mean([res["F1*-Measure"] for res in results])
    }
    return avg_results

def load_cleaned_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_shingles(text, k=5, features=None):
    words = text.split()
    synonyms = {"tv": ["television", "screen"], "led": ["light-emitting diode"]}
    # Vervang woorden met synoniemen
    words = [synonyms.get(word, [word])[0] for word in words]
    shingles = [' '.join(words[i:i + k]) for i in range(len(words) - k + 1)]
    if features:
        for key, value in features.items():
            if value:
                shingles.append(f"{key}:{value}")
    return shingles


def extract_model_words(text):
    return [word for word in text.split() if any(char.isdigit() for char in word)]

def generate_one_hot_vectors(products, shingles_vocab):
    shingle_to_index = {shingle: idx for idx, shingle in enumerate(shingles_vocab)}
    for product in products:
        shingles = product.get('shingles', [])
        one_hot = [0] * len(shingles_vocab)
        for shingle in shingles:
            if shingle in shingle_to_index:
                one_hot[shingle_to_index[shingle]] = 1
        product['one_hot'] = one_hot


def banding_method(products, num_bands, rows_per_band):
    buckets = [defaultdict(list) for _ in range(num_bands)]  # Eén dictionary per band

    for idx, product in enumerate(products):
        signature = product['minhash_signature']
        for band_idx in range(num_bands):
            start, end = band_idx * rows_per_band, (band_idx + 1) * rows_per_band
            band = tuple(signature[start:end])
            buckets[band_idx][band].append(idx)

    candidate_pairs = set()
    for bucket in buckets:
        for items in bucket.values():
            for i, j in combinations(items, 2):
                brand_i = products[i].get('brand', "Unknown")
                brand_j = products[j].get('brand', "Unknown")
                if brand_i == "Unknown" or brand_j == "Unknown" or brand_i == brand_j:
                    candidate_pairs.add((i, j))

    return candidate_pairs

def hierarchical_clustering(products, candidate_pairs, threshold):
    num_products = len(products)
    distance_matrix = np.full((num_products, num_products), 1.0)
    np.fill_diagonal(distance_matrix, 0.0)

    for a, b in candidate_pairs:
        similarity = msm_similarity(products[a], products[b])
        distance_matrix[a, b] = 1 - similarity
        distance_matrix[b, a] = 1 - similarity

    condensed_matrix = squareform(distance_matrix)
    Z = linkage(condensed_matrix, method='average')
    return fcluster(Z, t=threshold, criterion='distance')


def msm_similarity(product1, product2, weights=None):
    if weights is None:
        weights = {'shingles': 0.9, 'features': 0.1}

    if product1['brand'] != product2['brand']:
        return 1.0

    shingles1 = set(product1['shingles'])
    shingles2 = set(product2['shingles'])
    shingle_similarity = len(shingles1 & shingles2) / len(shingles1 | shingles2) if shingles1 | shingles2 else 0

    overlapping_keys = set(product1['featuresMap'].keys()) & set(product2['featuresMap'].keys())
    feature_similarity = sum(
        1 for key in overlapping_keys if product1['featuresMap'][key] == product2['featuresMap'][key]
    ) / len(overlapping_keys) if overlapping_keys else 0

    return weights['shingles'] * shingle_similarity + weights['features'] * feature_similarity

def evaluate_clusters(test_data, clusters, train_data):
    print(f"Aantal clusters: {len(set(clusters))}")
    print(f"Eerste paar clusters: {clusters[:10]}")

    # Maak een mapping tussen `train_data` en `test_data` op basis van unieke identificatoren
    train_to_test_index = {
        train['modelID']: idx
        for idx, train in enumerate(test_data)
        if 'modelID' in train
    }

    # Controleer of de mapping correct is
    print(f"Train-to-test mapping (eerste 5): {list(train_to_test_index.items())[:5]}")

    ground_truth = defaultdict(set)
    for idx, product in enumerate(test_data):
        ground_truth[product['modelID']].add(idx)

    detected_pairs = set()
    for cluster_id in set(clusters):
        cluster_members = [idx for idx, c_id in enumerate(clusters) if c_id == cluster_id]
        test_indices = [
            train_to_test_index.get(train['modelID'])
            for train in (train_data[idx] for idx in cluster_members)
            if train['modelID'] in train_to_test_index
        ]
        if len(test_indices) > 1:
            detected_pairs.update(combinations(test_indices, 2))

    # Controleer inhoud van detected pairs
    print(f"Detected pairs in test data: {len(detected_pairs)}")

    tp = sum(1 for a, b in detected_pairs if test_data[a]['modelID'] == test_data[b]['modelID'])
    total_comparisons = len(detected_pairs)
    total_duplicates = sum(len(indices) * (len(indices) - 1) // 2 for indices in ground_truth.values())

    print(f"TP: {tp}, Total Comparisons: {total_comparisons}, Total Duplicates: {total_duplicates}")

    pair_quality = tp / total_comparisons if total_comparisons > 0 else 0
    pair_completeness = tp / total_duplicates if total_duplicates > 0 else 0
    f1_star = (
        2 * pair_quality * pair_completeness / (pair_quality + pair_completeness)
        if pair_quality + pair_completeness > 0 else 0
    )

    return {"Pair Quality": pair_quality, "Pair Completeness": pair_completeness, "F1*-Measure": f1_star}

# Main workflow
cleaned_path = "/Users/josephine/PycharmProjects/ComputerScienceTwo/cleaned-data.json"
data = load_cleaned_data(cleaned_path)

# Flatten products
products = [item for sublist in data.values() for item in sublist]

# Globale shingles_vocab gebaseerd op de volledige dataset
global_shingles_vocab = sorted(
    set(shingle for product in products for shingle in generate_shingles(product['title']))
)

# MinHash parameters
num_hashes = 9
max_val = len(global_shingles_vocab) * 10
hash_functions = [(random.randint(1, max_val), random.randint(0, max_val)) for _ in range(num_hashes)]

# Bootstrapping evaluatie
num_bootstraps = 5
final_results = evaluate_with_bootstrapping(products, num_bootstraps=num_bootstraps)
print("Gemiddelde resultaten:", final_results)


