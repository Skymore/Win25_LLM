#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "===== 开始生成改进版Kaggle提交文件 ====="
echo "当前工作目录: $(pwd)"

# 运行改进版预测脚本
echo -e "\n===== 运行改进版预测脚本 ====="
python improved_kaggle.py

# 验证提交文件
echo -e "\n===== 验证提交文件 ====="
python verify_submission.py "improved_kaggle_submission.csv"

echo -e "\n===== 完成 ====="
echo "提交文件已生成: $SCRIPT_DIR/improved_kaggle_submission.csv"
echo "请将此文件上传到Kaggle竞赛页面。" 