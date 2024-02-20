#!/bin/bash

# 当任何语句的执行结果不是true时应该终止shell脚本
set -e

# 函数定义，用于执行训练命令
train() {
    python train_human.py --dataset "$1"
}

# 针对每个fold运行训练命令
# 42系列的fold
train human/raw/42/fold_1
train human/raw/42/fold_2
train human/raw/42/fold_3
train human/raw/42/fold_4
train human/raw/42/fold_5

# 52系列的fold
train human/raw/52/fold_1
train human/raw/52/fold_2
train human/raw/52/fold_3
train human/raw/52/fold_4
train human/raw/52/fold_5

# 62系列的fold
train human/raw/62/fold_1
train human/raw/62/fold_2
train human/raw/62/fold_3
train human/raw/62/fold_4
train human/raw/62/fold_5

echo "所有fold的训练已完成。"
