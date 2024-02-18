# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse
from sklearn import metrics
from metrics import *
from utils import *
from sklearn.metrics import average_precision_score,precision_recall_curve,auc

from metrics import get_cindex, get_rm2
from dataset import *
from ParaCPI import MGraphDTA
from utils import *
from metrics import *
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
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
def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:
        data.y = data.y.long()  
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred, data.y)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(data.y.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

    pred = np.concatenate(pred_list, axis=0)
    # print(pred)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    # print(pred_cls)
    label = np.concatenate(label_list, axis=0)
    train_data = {'pred': pred.tolist(), 'pred_cls': pred_cls.tolist()}
    Save_to_Csv(data=train_data, file_name='pred', Save_format='csv', Save_type='col')
    acc = accuracy(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    AUC = auc_score(label, pred)

    
    precision1,reacall1,r = precision_recall_curve(label, pred_cls)
    aupr = auc(reacall1, precision1)
    print(precision1.tolist())
    print(reacall1.tolist())
    print(aupr)
    fpr, tpr, threshold = metrics.roc_curve(label, pred)
    Auc = auc(fpr, tpr)
    print(fpr.tolist())
    print(tpr.tolist())
    print(Auc)
    

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, pre, rec, AUC

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', default='human', help='human or celegans')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    args = parser.parse_args()

    data_root = "data"
    DATASET = args.dataset
    model_path = args.model_path

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, types='test')
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    device = torch.device('cuda:0')
    model = MGraphDTA(1, 1,1,filter_num=32, out_dim=2).to(device)

    criterion = nn.CrossEntropyLoss()
    load_model_dict(model, model_path)

    test_loss, test_acc, test_pre, test_rec, test_auc  = val(model, criterion, test_loader, device)
    msg = "test_loss-%.4f, test_acc-%.4f, test_pre-%.4f, test_rec-%.4f, test_auc-%.4f" % (test_loss, test_acc, test_pre, test_rec, test_auc)

    print(msg)

if __name__ == "__main__":
    main()
