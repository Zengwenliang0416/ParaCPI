#!/bin/bash

# Exit if any command fails
set -e

# Define a function for training a specific dataset fold
train_fold() {
    python train_celegans.py --dataset "$1"
}

# Train all folds for dataset series 42
train_fold celegans/raw/42/fold_1
train_fold celegans/raw/42/fold_2
train_fold celegans/raw/42/fold_3
train_fold celegans/raw/42/fold_4
train_fold celegans/raw/42/fold_5

# Train all folds for dataset series 52
train_fold celegans/raw/52/fold_1
train_fold celegans/raw/52/fold_2
train_fold celegans/raw/52/fold_3
train_fold celegans/raw/52/fold_4
train_fold celegans/raw/52/fold_5

# Train all folds for dataset series 62
train_fold celegans/raw/62/fold_1
train_fold celegans/raw/62/fold_2
train_fold celegans/raw/62/fold_3
train_fold celegans/raw/62/fold_4
train_fold celegans/raw/62/fold_5

echo "Training on all C. elegans dataset folds is complete."
