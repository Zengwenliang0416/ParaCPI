import os
import pandas as pd
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv('data.csv')

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
        test_data.to_csv(os.path.join(fold_dir, 'data_test.csv'), index=False)

print("The datasets for 5-fold cross-validation have been split and saved in respective folders by random state.")
