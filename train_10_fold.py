# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
from sklearn.model_selection import KFold
from metrics import accuracy, precision, auc_score, recall
from dataset import *
from model3 import MGraphDTA
from utils import *
from log.train_logger import TrainLogger

def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    #maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce
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
    auc = auc_score(label, pred)
    roce1 = round(getROCE(pred_cls, label, 0.5), 2)
    roce2 = round(getROCE(pred_cls, label,  1), 2)
    roce3 = round(getROCE(pred_cls, label,  2), 2)
    roce4 = round(getROCE(pred_cls, label,  5), 2)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, pre, rec, auc,roce1,roce2,roce3,roce4

def main():
    parser = argparse.ArgumentParser()

    # Add argument BindingDB
    parser.add_argument('--dataset', default='DUDE', help='human or celegans') #required=True,
    parser.add_argument('--save_model', default='True', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    
    kfold = KFold(n_splits=10, shuffle=True)
    train_set = GNNDataset(fpath, types='train')
    test_set = GNNDataset(fpath, types='test')

    for fold, (train_index, val_index) in enumerate(kfold.split(train_set)):
        # 获取训练集和验证集
        train_set_fold = Subset(train_set, train_index)
        val_set_fold = Subset(test_set, val_index)
        
        # 将训练集和验证集放入DataLoader中
        train_loader_fold = DataLoader(train_set_fold, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader_fold = DataLoader(val_set_fold, batch_size=params['batch_size'], shuffle=False, num_workers=4)
        device = torch.device('cuda:0')
        epochs = 1000
        steps_per_epoch = 10
        n = len(train_loader_fold)
        model = MGraphDTA(epochs, steps_per_epoch,n,filter_num=32, out_dim=2).to(device)

        num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader_fold))

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()

        global_step = 0
        global_epoch = 0

        running_loss = AverageMeter()
        model.train()
        # 训练模型
        for i in range(num_iter):
          for data in train_loader_fold:
            global_step += 1
            data.y = data.y.long()
            data = data.to(device)
            pred = model(data,i)
            loss = criterion(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                test_loss, test_acc, test_pre, test_rec, test_auc,roce1,roce2,roce3,roce4 = val(model, criterion, val_loader_fold, device)

                msg = "epoch-%d, loss-%.3f, test_acc-%.3f, test_pre-%.3f, test_rec-%.3f, test_auc-%.3f, roce1-%.3f, roce2-%.3f, roce3-%.3f, roce4-%.3f" % (global_epoch, test_loss, test_acc, test_pre, test_rec, test_auc,roce1,roce2,roce3,roce4)
                logger.info(msg)

                if save_model and test_auc >=0.95:
                    save_model_dict(model, logger.get_model_dir(), msg)



if __name__ == "__main__":
    main()
