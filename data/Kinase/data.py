import pandas as pd

# 定义列名
column_names = ['compound_iso_smiles', 'target_sequence', 'affinity']

# 读取训练集文本文件，添加列名
data_train = pd.read_csv('kinase_train_original.txt', sep=' ', header=None, names=column_names)

# 保存DataFrame到CSV文件，没有索引列
data_train.to_csv('data_train.csv', index=False)

# 读取测试集文本文件，添加列名
data_test = pd.read_csv('kinase_test.txt', sep=' ', header=None, names=column_names)

# 保存DataFrame到CSV文件，没有索引列
data_test.to_csv('data_test.csv', index=False)

import os
import pandas as pd
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv('data_train.csv')

# Prepare 5-fold cross-validation
n_splits = 5
random_states = [42, 52, 62]

for state in random_states:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=state)

    # Loop over each fold
    for fold_number, (train_index, test_index) in enumerate(kf.split(data), start=1):
        # Create directory name for the fold, including the random_state
        fold_dir = os.path.join(str(state), f'fold_{fold_number}')
        os.makedirs(fold_dir, exist_ok=True)  # Create directory if it does not exist

        # Split the dataset into training and testing sets
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Save the training and testing sets as CSV files
        train_data.to_csv(os.path.join(fold_dir, 'data_train.csv'), index=False)
        test_data.to_csv(os.path.join(fold_dir, 'data_val.csv'), index=False)

print("The datasets for 5-fold cross-validation have been split and saved in respective folders by random state.")

