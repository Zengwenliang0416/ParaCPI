#!/bin/bash

read -p "Enter commit message: " message
read -p "Enter branch name (default is master): " branch

branch=${branch:-master}  # 如果没有输入分支名，使用master作为默认值

git add .
git commit -m "$message"
git push origin "$branch"

echo "Commit and push have been executed."
