#!/bin/bash

# Exit immediately if any command exits with a non-zero status.
set -e

# Function to preprocess a single dataset fold.
preprocess_fold() {
    python preprocessing_celegans.py --dataset "$1"
}

# Preprocess all the folds for the 42 series.
preprocess_fold data/celegans/raw/42/fold_1
preprocess_fold data/celegans/raw/42/fold_2
preprocess_fold data/celegans/raw/42/fold_3
preprocess_fold data/celegans/raw/42/fold_4
preprocess_fold data/celegans/raw/42/fold_5

# Preprocess all the folds for the 52 series.
preprocess_fold data/celegans/raw/52/fold_1
preprocess_fold data/celegans/raw/52/fold_2
preprocess_fold data/celegans/raw/52/fold_3
preprocess_fold data/celegans/raw/52/fold_4
preprocess_fold data/celegans/raw/52/fold_5

# Preprocess all the folds for the 62 series.
preprocess_fold data/celegans/raw/62/fold_1
preprocess_fold data/celegans/raw/62/fold_2
preprocess_fold data/celegans/raw/62/fold_3
preprocess_fold data/celegans/raw/62/fold_4
preprocess_fold data/celegans/raw/62/fold_5

echo "Preprocessing for all C. elegans dataset folds is complete."

