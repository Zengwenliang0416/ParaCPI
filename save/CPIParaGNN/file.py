import glob
import os
import shutil

# 定义搜索的起始目录
start_dir = '.'  # 当前目录

# 定义搜索模式
pattern = '*_celegans-NE/raw/*/fold_*/log/train/Train.log'

# 使用glob找到所有匹配的文件
files = glob.glob(os.path.join(start_dir, pattern), recursive=True)

# 循环处理每个找到的文件
for file_path in files:
    # 提取fold部分
    fold_part = file_path.split(os.sep)[4]  # 根据目录结构调整索引
    # 定义新路径
    new_path = os.path.join('celegans-NE', file_path.split(os.sep)[3], fold_part, 'log', 'train', 'Train.log')
    # 创建新目录
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    # 移动文件
    shutil.copy(file_path, new_path)
    print(f"复制文件：{file_path} 到 {new_path}")

print("所有文件复制完成。")



# 定义搜索的起始目录
start_dir = '.'  # 当前目录

# 定义搜索模式
pattern = '*_human-NE/raw/*/fold_*/log/train/Train.log'

# 使用glob找到所有匹配的文件
files = glob.glob(os.path.join(start_dir, pattern), recursive=True)

# 循环处理每个找到的文件
for file_path in files:
    # 提取fold部分
    fold_part = file_path.split(os.sep)[4]  # 根据目录结构调整索引
    # 定义新路径
    new_path = os.path.join('human-NE', file_path.split(os.sep)[3], fold_part, 'log', 'train', 'Train.log')
    # 创建新目录
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    # 移动文件
    shutil.copy(file_path, new_path)
    print(f"复制文件：{file_path} 到 {new_path}")

print("所有文件复制完成。")
