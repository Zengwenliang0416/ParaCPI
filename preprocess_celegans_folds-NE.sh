#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Function to preprocess a single dataset fold.
preprocess_fold() {
    python preprocessing_celegans-NE.py --dataset "$1"
}

# Function to preprocess all folds for a given series.
preprocess_series() {
    local series="$1"
    for fold in {1..5}; do
        preprocess_fold "data/celegans-NE/raw/${series}/fold_${fold}"
    done
}

# Array of dataset series
dataset_series=(42 52 62)

# Loop through each dataset series and preprocess all folds
for series in "${dataset_series[@]}"; do
    preprocess_series "${series}"
done

echo "Preprocessing for all C. elegans dataset folds is complete."
