#!/bin/bash

# Exit if any command fails
set -e

# Define a function for training a specific dataset fold
train_fold() {
    python train_celegans.py --dataset "$1" --modelName "$2"
}

# Function to train all folds for a given dataset series
train_series() {
    local series="$1"
    local model="$2"
    for fold in {1..5}; do
        train_fold "celegans-NE/raw/${series}/fold_${fold}" "${model}"
    done
}

# Define the model name
modelName="CPIGRB"

# Array of dataset series
dataset_series=(42 52 62)

# Loop through each dataset series and train all folds
for series in "${dataset_series[@]}"; do
    train_series "${series}" "${modelName}"
done

echo "Training on all C. elegans dataset folds is complete."
