#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "===== 开始生成Kaggle提交文件 ====="
echo "当前工作目录: $(pwd)"

# 运行预测脚本
echo -e "\n===== 运行预测脚本 ====="
python kaggle_submission.py

# # 验证提交文件
# echo -e "\n===== 验证提交文件 ====="
# python verify_submission.py

echo -e "\n===== 完成 ====="
echo "提交文件已生成: $SCRIPT_DIR/high_rank_direct_2.csv"
echo "请将此文件上传到Kaggle竞赛页面。" 