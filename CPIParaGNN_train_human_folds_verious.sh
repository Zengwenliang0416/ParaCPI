#!/bin/bash

# Exit if any command fails
set -e

# Define a function for training a specific dataset fold
train_fold() {
    python train_human.py --dataset "$1" --modelName "$2"
}

# Function to train all folds for a given dataset series
train_series() {
    for fold in {1..5}; do
        train_fold "human/raw/$1/fold_$fold" CPIParaGNN
    done
}

# Array of dataset series
dataset_series=(42 52 62)

# Loop through each dataset series and train all folds
for series in "${dataset_series[@]}"; do
    train_series "$series"
done

echo "Training on all human dataset folds is complete."
