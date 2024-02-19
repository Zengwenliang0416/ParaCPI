import csv
import random

# 读取CSV文件
with open('data_train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

# 打乱数据顺序
random.seed(443)
random.shuffle(data)

# 将数据分成五个折
num_folds = 5
fold_size = len(data) // num_folds
folds = []
for i in range(num_folds):
    fold = data[i*fold_size : (i+1)*fold_size]
    folds.append(fold)
fields = ['compound_iso_smiles', 'target_sequence','affinity']
# 生成训练集和测试集
for i in range(num_folds):
    test_data = folds[i]
    train_data = []
    for j in range(num_folds):
        if j != i:
            train_data += folds[j]
    # 将训练集和测试集存储为CSV文件
    with open(f'data_train{i+1}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(train_data)
    with open(f'data_test{i+1}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(test_data)