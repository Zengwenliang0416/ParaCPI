#!/bin/bash

# Exit if any command fails
set -e

# Define a function for training a specific dataset fold
train_fold() {
    python train_celegans.py --dataset "$1" --modelName "$2"
}

# Train all folds for dataset series 42
train_fold celegans/raw/42/fold_1 CPIDSCNN
train_fold celegans/raw/42/fold_2 CPIDSCNN
train_fold celegans/raw/42/fold_3 CPIDSCNN
train_fold celegans/raw/42/fold_4 CPIDSCNN
train_fold celegans/raw/42/fold_5 CPIDSCNN

# Train all folds for dataset series 52
train_fold celegans/raw/52/fold_1 CPIDSCNN
train_fold celegans/raw/52/fold_2 CPIDSCNN
train_fold celegans/raw/52/fold_3 CPIDSCNN
train_fold celegans/raw/52/fold_4 CPIDSCNN
train_fold celegans/raw/52/fold_5 CPIDSCNN

# Train all folds for dataset series 62
train_fold celegans/raw/62/fold_1 CPIDSCNN
train_fold celegans/raw/62/fold_2 CPIDSCNN
train_fold celegans/raw/62/fold_3 CPIDSCNN
train_fold celegans/raw/62/fold_4 CPIDSCNN
train_fold celegans/raw/62/fold_5 CPIDSCNN

# Train all folds for dataset series 42
train_fold celegans/raw/42/fold_1 CPIGRB
train_fold celegans/raw/42/fold_2 CPIGRB
train_fold celegans/raw/42/fold_3 CPIGRB
train_fold celegans/raw/42/fold_4 CPIGRB
train_fold celegans/raw/42/fold_5 CPIGRB

# Train all folds for dataset series 52
train_fold celegans/raw/52/fold_1 CPIGRB
train_fold celegans/raw/52/fold_2 CPIGRB
train_fold celegans/raw/52/fold_3 CPIGRB
train_fold celegans/raw/52/fold_4 CPIGRB
train_fold celegans/raw/52/fold_5 CPIGRB

# Train all folds for dataset series 62
train_fold celegans/raw/62/fold_1 CPIGRB
train_fold celegans/raw/62/fold_2 CPIGRB
train_fold celegans/raw/62/fold_3 CPIGRB
train_fold celegans/raw/62/fold_4 CPIGRB
train_fold celegans/raw/62/fold_5 CPIGRB


# Train all folds for dataset series 42
train_fold celegans/raw/42/fold_1 CPIParaGNN
train_fold celegans/raw/42/fold_2 CPIParaGNN
train_fold celegans/raw/42/fold_3 CPIParaGNN
train_fold celegans/raw/42/fold_4 CPIParaGNN
train_fold celegans/raw/42/fold_5 CPIParaGNN

# Train all folds for dataset series 52
train_fold celegans/raw/52/fold_1 CPIParaGNN
train_fold celegans/raw/52/fold_2 CPIParaGNN
train_fold celegans/raw/52/fold_3 CPIParaGNN
train_fold celegans/raw/52/fold_4 CPIParaGNN
train_fold celegans/raw/52/fold_5 CPIParaGNN

# Train all folds for dataset series 62
train_fold celegans/raw/62/fold_1 CPIParaGNN
train_fold celegans/raw/62/fold_2 CPIParaGNN
train_fold celegans/raw/62/fold_3 CPIParaGNN
train_fold celegans/raw/62/fold_4 CPIParaGNN
train_fold celegans/raw/62/fold_5 CPIParaGNN


echo "Training on all C. elegans dataset folds is complete."
