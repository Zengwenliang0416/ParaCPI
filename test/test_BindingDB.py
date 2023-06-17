# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import argparse
from sklearn.metrics import average_precision_score,precision_recall_curve,auc

from metrics import get_cindex, get_rm2
from dataset import *
from model_BindingDB import MGraphDTA
from utils import *
from metrics import *

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
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    AUC = auc_score(label, pred)
    tpr,fpr,_ = precision_recall_curve(label, pred_cls)
    prc = auc(fpr,tpr)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, pre, rec, AUC, prc

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', default='BindingDB', help='human or celegans')
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

    test_loss, test_acc, test_pre, test_rec, test_auc, test_prc  = val(model, criterion, test_loader, device)
    msg = "test_loss-%.3f, test_acc-%.3f, test_pre-%.3f, test_rec-%.3f, test_auc-%.3f, test_prc-%.3f" % (test_loss, test_acc, test_pre, test_rec, test_auc,test_prc)

    print(msg)

if __name__ == "__main__":
    main()
