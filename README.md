# Product Duplicate Detection

This repository provides a Python implementation for detecting duplicate products in e-commerce datasets. The project consists of two main components: **data preprocessing** and **duplicate detection**. The implementation combines methods such as MinHash, Locality Sensitive Hashing (LSH), Multi-component Similarity Method (MSM), and hierarchical clustering to balance efficiency and accuracy in large datasets.

---

## Overview

The goal of this project is to detect duplicate products from e-commerce datasets that contain product titles, features, and additional attributes. The workflow includes:
1. Preprocessing product data to generate structured inputs for duplicate detection.
2. Using a combination of LSH and hierarchical clustering with a custom similarity metric (MSM) to identify duplicates efficiently and effectively.

---

## Features

### Data Preprocessing
Normalizes product titles by:
  - Converting text to lowercase.
  - Standardizing terms like "inch" and "hz."
  - Removing special characters.
- Maps brand variations to standardized names.
- Extracts brand names from titles or features and handles missing brands.

### Duplicate Detection
MinHash-based LSH for scalable candidate pair generation.
- Multi-component Similarity Method (MSM) combining:
  - **Shingle-based title similarity.**
  - **Feature-based key-value similarity.**
- Hierarchical clustering for adaptive grouping of products.

---

## Data Preprocessing Functions

### `normalize_text(text)`
Converts text to lowercase and applies standard transformations, such as:
- Removing special characters.
- Replacing multiple spaces with single spaces.
- Standardizing terms like "inch" and "hz."

---

### `map_brand_variations(brand)`
Maps variations of brand names (e.g., "lg electronics" → "LG") to their standardized names using a predefined dictionary.

---

### `clean_data(file_path, output_path, brand_list)`
Cleans and normalizes the dataset:
- Extracts brand names from product titles or feature maps.
- Assigns the value "Overig" (Other) to products missing a brand.
- Saves the cleaned dataset as a JSON file.

---

## Duplicate Detection Functions

### `minhash_signature(one_hot, hash_functions, max_val)`
Generates a MinHash signature for a one-hot vector using a list of hash functions.

---

### `bootstrap_data(data, seed=None)`
Splits the dataset into training and test sets using bootstrapping:
- Selects approximately 63% of the data for training and the rest for testing.

---

### `evaluate_with_bootstrapping(products, num_bootstraps=5, threshold=0.5)`
Runs the duplicate detection pipeline on multiple bootstrap samples:
- Calculates average metrics (e.g., Pair Quality, Pair Completeness, F1*-Measure) across the samples.

---

### `load_cleaned_data(file_path)`
Loads a cleaned JSON dataset from the specified file.

---

### `generate_shingles(text, k=5, features=None)`
Creates shingles (overlapping n-grams) from product titles:
- Optionally incorporates key-value pairs from the product's feature map.

---

### `extract_model_words(text)`
Extracts model-specific words (e.g., "1080p," "120hz") from product titles by identifying words containing numeric characters.

---

### `generate_one_hot_vectors(products, shingles_vocab)`
Converts shingles of each product into a one-hot encoded vector based on a global vocabulary.

---

### `banding_method(products, num_bands, rows_per_band)`
Implements Locality Sensitive Hashing (LSH) by:
- Grouping MinHash signatures into bands.
- Generating candidate pairs of products based on similar bands.

---

### `hierarchical_clustering(products, candidate_pairs, threshold)`
Clusters products based on pairwise distances computed using the Multi-component Similarity Method (MSM):
- Performs hierarchical agglomerative clustering.
- Cuts the dendrogram at the specified threshold to form clusters.

---

### `msm_similarity(product1, product2, weights=None)`
Calculates the similarity between two products using a weighted combination of:
- Shingle similarity: Overlap of n-grams in product titles.
- Feature similarity: Overlap of key-value pairs in feature maps.
- Optionally incorporates brand similarity.

---

### `evaluate_clusters(test_data, clusters, train_data)`
Evaluates clustering results using metrics such as Pair Quality, Pair Completeness, and F1*-Measure.

---
