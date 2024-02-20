# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score, matthews_corrcoef
import argparse
from metrics import precision, auc_score, recall
from dataset import *
from ParaCPI import ParaCPI
from CPIDSCNN import CPIDSCNN
from CPIGRB import CPIGRB
from CPIParaGNN import CPIParaGNN
from CPINE import CPINE
from utils import *
from log.train_logger import TrainLogger
from preprocess.preprocessing_celegans import *


def getROCE(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x: x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    # maxIndexs = []
    for x in predList:
        if (targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if (fp1 > ((roceRate * n) / 100)):
                break
    roce = (tp1 * n) / (p * fp1)
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

    precision1, reacall1, r = precision_recall_curve(label, pred_cls)
    aupr = round(auc(reacall1, precision1), 3)
    # Calculate MCC and F1
    mcc = round(matthews_corrcoef(label, pred_cls), 3)
    f1 = round(f1_score(label, pred_cls), 3)
    pre = round(precision(label, pred_cls), 3)
    rec = round(recall(label, pred_cls), 3)
    Auc = round(auc_score(label, pred), 3)

    # pre = precision(label, pred_cls)
    # rec = recall(label, pred_cls)
    # auc = auc_score(label, pred)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, pre, rec, Auc, aupr, mcc, f1


def main():
    parser = argparse.ArgumentParser()
    # 对比实验
    # python train_human.py --dataset human/raw/42/fold_1
    # python train_human.py --dataset human/raw/42/fold_2
    # python train_human.py --dataset human/raw/42/fold_3
    # python train_human.py --dataset human/raw/42/fold_4
    # python train_human.py --dataset human/raw/42/fold_5

    # python train_human.py --dataset human/raw/52/fold_1
    # python train_human.py --dataset human/raw/52/fold_2
    # python train_human.py --dataset human/raw/52/fold_3
    # python train_human.py --dataset human/raw/52/fold_4
    # python train_human.py --dataset human/raw/52/fold_5

    # python train_human.py --dataset human/raw/62/fold_1
    # python train_human.py --dataset human/raw/62/fold_2
    # python train_human.py --dataset human/raw/62/fold_3
    # python train_human.py --dataset human/raw/62/fold_4
    # python train_human.py --dataset human/raw/62/fold_5

    # 消融实验
    # python train_human.py --dataset human/raw/42/fold_1 --model CPIDSCNN
    # python train_human.py --dataset human/raw/42/fold_2 --model CPIDSCNN
    # python train_human.py --dataset human/raw/42/fold_3 --model CPIDSCNN
    # python train_human.py --dataset human/raw/42/fold_4 --model CPIDSCNN
    # python train_human.py --dataset human/raw/42/fold_5 --model CPIDSCNN

    # python train_human.py --dataset human/raw/52/fold_1 --model CPIGRB
    # python train_human.py --dataset human/raw/52/fold_2 --model CPIGRB
    # python train_human.py --dataset human/raw/52/fold_3 --model CPIGRB
    # python train_human.py --dataset human/raw/52/fold_4 --model CPIGRB
    # python train_human.py --dataset human/raw/52/fold_5 --model CPIGRB

    # python train_human.py --dataset human/raw/62/fold_1 --model CPIParaGNN
    # python train_human.py --dataset human/raw/62/fold_2 --model CPIParaGNN
    # python train_human.py --dataset human/raw/62/fold_3 --model CPIParaGNN
    # python train_human.py --dataset human/raw/62/fold_4 --model CPIParaGNN
    # python train_human.py --dataset human/raw/62/fold_5 --model CPIParaGNN

    parser.add_argument('--modelName', default='ParaCPI', help='GPCR or Kinase')  # required=True,
    parser.add_argument('--model', default='CPIDSCNN', help='GPCR or Kinase')  # required=True,
    parser.add_argument('--dataset', default='human/raw/42/fold_1', help='GPCR or Kinase')  # required=True,
    parser.add_argument('--save_model', default='True', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    args = parser.parse_args()

    params = dict(data_root="data", save_dir="save", dataset=args.dataset, save_model=args.save_model, lr=args.lr,
                  batch_size=args.batch_size, modelName=args.modelName)

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    fpath = os.path.join(data_root, DATASET)
    GNNDataset(root=fpath)
    train_set = GNNDataset(fpath, types='train')
    test_set = GNNDataset(fpath, types='test')

    logger.info(f"Number of train: {len(train_set)}")
    logger.info(f"Number of test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False, num_workers=4)

    device = torch.device('cuda:0')
    epochs = 100
    steps_per_epoch = 10
    n = len(train_loader)

    if (args.modelName == "ParaCPI"):
        model = ParaCPI(epochs, steps_per_epoch, n, filter_num=32, out_dim=2).to(device)
    elif (args.modelName == "CPINE"):
        model = CPINE(epochs, steps_per_epoch, n, filter_num=32, out_dim=2).to(device)
    elif (args.modelName == "CPIDSCNN"):
        model = CPIDSCNN(epochs, steps_per_epoch, n, filter_num=32, out_dim=2).to(device)
    elif (args.modelName == "CPIGRB"):
        model = CPIGRB(epochs, steps_per_epoch, n, filter_num=32, out_dim=2).to(device)
    elif (args.modelName == "CPIParaGNN"):
        model = CPIParaGNN(epochs, steps_per_epoch, n, filter_num=32, out_dim=2).to(device)
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    global_epoch = 0

    running_loss = AverageMeter()
    torch.cuda.empty_cache()
    model.train()

    for i in range(num_iter):
        for data in train_loader:
            global_step += 1
            data.y = data.y.long()
            data = data.to(device)
            pred = model(data, i)
            loss = criterion(pred, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                test_loss, test_pre, test_rec, test_auc, test_aupr, test_mcc, test_f1 = val(model, criterion,
                                                                                            test_loader, device)

                msg = "%s_epoch-%d_loss-%.3f_pre-%.3f_rec-%.3f_auc-%.3f_aupr-%.3f_mcc-%.3f_f1-%.3f" % (
                    params.get("dataset").replace("/", "-"), global_epoch, test_loss, test_pre, test_rec, test_auc,
                    test_aupr, test_mcc, test_f1)
                logger.info(msg)

                if save_model and global_epoch > epochs - 10:
                    save_model_dict(model, logger.get_model_dir(), msg)


if __name__ == "__main__":
    main()
