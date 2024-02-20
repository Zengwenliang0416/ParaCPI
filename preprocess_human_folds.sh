#!/bin/bash

# 确保脚本在遇到错误时终止执行
set -e

# 定义函数进行预处理
preprocess() {
    python preprocessing_human.py --dataset "$1"
}

# 42系列的fold
preprocess data/human/raw/42/fold_2
preprocess data/human/raw/42/fold_3
preprocess data/human/raw/42/fold_4
preprocess data/human/raw/42/fold_5

# 52系列的fold
preprocess data/human/raw/52/fold_1
preprocess data/human/raw/52/fold_2
preprocess data/human/raw/52/fold_3
preprocess data/human/raw/52/fold_4
preprocess data/human/raw/52/fold_5

# 62系列的fold
preprocess data/human/raw/62/fold_1
preprocess data/human/raw/62/fold_2
preprocess data/human/raw/62/fold_3
preprocess data/human/raw/62/fold_4
preprocess data/human/raw/62/fold_5

echo "预处理完成。"

