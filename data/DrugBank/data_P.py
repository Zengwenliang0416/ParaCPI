import csv
import random
import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
# 读取CSV文件
def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....}
    # 列名即为CSV中数据对应的列名， 数据为一个列表

    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储

    # 默认存储在当前路径下,可根据自己需要进行更改

    Name = []
    times = 0

    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))

            times += 1

        Pd_data = pd.DataFrame(columns=Name, data=Data)

    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))

            times += 1

        Pd_data = pd.DataFrame(index=Name, data=Data)

    if Save_format == 'csv':
        Pd_data.to_csv('./' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('./' + file_name + '.xls', encoding='utf-8')
# data = pd.read_csv('./Davis.txt')
# data.to_csv('./data.csv')

data = pd.read_csv('DrugBank.csv')
com_num = [i for i in range(6655)]
pro_num = [i for i in range(4294)]
compound = list(set(data['compound_iso_smiles']))
protein = list(set(data['target_sequence']))
compound_dic = dict(zip(compound,com_num))
protein_dic = dict(zip(protein,pro_num))
c = []
p= []
x = list(data['compound_iso_smiles'])
# l = len(data['compound_iso_smiles'])
for i in range(len(data['compound_iso_smiles'])):
    c.append(compound_dic[data['compound_iso_smiles'][i]])
    p.append(protein_dic[data['target_sequence'][i]])

train_data = {'compound_iso_smiles': list(data['compound_iso_smiles']), 'target_sequence': list(data['target_sequence']),
              'affinity': list(data['affinity']), 'com_num': c, 'pro_num': p}
Save_to_Csv(data=train_data, file_name='DrugBank' +'_cold', Save_format='csv', Save_type='col')

pid_length = max(p)
did_length = max(c)
x = list(range(did_length)), int(did_length/5)
did_sample = random.sample(list(range(did_length)), int(did_length/5))
pid_sample = random.sample(list(range(pid_length)), int(pid_length/5))

print(did_sample)
drug = list(data['compound_iso_smiles'])
protein =list(data['target_sequence'])
affinity = list(data['affinity'])

train_drug, train_protein, train_affinity, train_pid, train_did = [], [], [], [], []
test_drug, test_protein, test_affinity, test_pid, test_did = [], [], [], [], []


for i in range(len(drug)):
    if p[i] in pid_sample:
        test_drug.append(drug[i])
        test_protein.append(protein[i])
        test_affinity.append(affinity[i])
        test_pid.append(p[i])
        test_did.append(c[i])
    else:
        train_drug.append(drug[i])
        train_protein.append(protein[i])
        train_affinity.append(affinity[i])
        train_pid.append(p[i])
        train_did.append(c[i])

print(len(train_drug))
print(len(test_drug))


train_data = {'compound_iso_smiles': train_drug, 'target_sequence': train_protein, 'affinity': train_affinity}
Save_to_Csv(data=train_data, file_name='DrugBank' +'_protein_cold_'+'train', Save_format='csv', Save_type='col')
test_data = {'compound_iso_smiles': test_drug, 'target_sequence': test_protein, 'affinity': test_affinity}
Save_to_Csv(data=test_data, file_name= 'DrugBank' +'_protein_cold_'+'test', Save_format='csv', Save_type='col')

# Load the dataset
data = pd.read_csv('DrugBank_protein_cold_train.csv')

# Prepare 5-fold cross-validation
n_splits = 5
random_states = [42, 52, 62]

for state in random_states:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=state)

    # Loop over each fold
    for fold_number, (train_index, test_index) in enumerate(kf.split(data), start=1):
        # Create directory name for the fold, including the random_state
        fold_dir = os.path.join("P",str(state), f'fold_{fold_number}')
        os.makedirs(fold_dir, exist_ok=True)  # Create directory if it does not exist

        # Split the dataset into training and testing sets
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Save the training and testing sets as CSV files
        train_data.to_csv(os.path.join(fold_dir, 'data_train.csv'), index=False)
        test_data.to_csv(os.path.join(fold_dir, 'data_test.csv'), index=False)

print("The datasets for 5-fold cross-validation have been split and saved in respective folders by random state.")
