import pandas as pd
import numpy as np
data_test = pd.read_csv('./kinase_test.txt')
data_test.to_csv('./data_test.csv')
data_train = pd.read_csv('./kinase_train_original.txt')
data_train.to_csv('./data_train.csv')


# train = pd.read_csv('./train.csv',usecols=['SMILES','Target Sequence','Label'])
# val = pd.read_csv('./val.csv',usecols=['SMILES','Target Sequence','Label'])
# test = pd.read_csv('./test.csv',usecols=['SMILES','Target Sequence','Label'])
# merged_df = pd.concat([train,val,test])
# merged_df = merged_df.rename(columns={'SMILES':'compoud_iso_smiles','Target Sequence':'target_sequence','Label':'affinity'})
# merged_df.insert(0,'index',range(0,len((merged_df))))
# merged_df.to_csv('../raw/data.csv',index=False)
#
# # # 设置随机数生成器种子，以确保结果可重复
# # np.random.seed(1)
# #
# # # 生成随机索引
# # idx = np.random.permutation(len(merged_df))
# #
# # # 计算数据集大小
# # data_size = len(merged_df)
# #
# # # 划分比例
# # train_ratio = 0.8
# # val_ratio = 0.1
# # test_ratio = 0.1
# #
# # # 计算每个数据集的大小
# # train_size = int(data_size * train_ratio)
# # val_size = int(data_size * val_ratio)
# # test_size = int(data_size * test_ratio)
# #
# # # 使用随机索引划分数据集
# # train_df = merged_df.iloc[idx[:train_size]]
# # val_df = merged_df.iloc[idx[train_size:train_size+val_size]]
# # test_df = merged_df.iloc[idx[train_size+val_size:]]
# #
# # # 将每个数据集存储为CSV文件
# # train_df.to_csv('data_train.csv', index=False)
# # val_df.to_csv('data_val.csv', index=False)
# # test_df.to_csv('data_test.csv', index=False)
# #
# #
# #
