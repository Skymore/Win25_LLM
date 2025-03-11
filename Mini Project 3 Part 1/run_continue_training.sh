#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "===== 开始继续微调模型 ====="
echo "当前工作目录: $(pwd)"

# 运行继续训练脚本
echo -e "\n===== 运行继续训练脚本 ====="
python continue_training.py

# 训练完成后，运行预测脚本
echo -e "\n===== 训练完成，运行预测脚本 ====="
echo "请修改以下变量来指向新的模型路径："
echo "修改 'model_path' 变量为: \"$SCRIPT_DIR/lora_continue_training/continue_training_high_rank_direct\""

# 为了方便用户使用新模型生成提交文件，自动修改kaggle_submission.py中的模型路径
sed -i "s|model_path = os.path.join(script_dir, \"lora_experiments/high_rank_direct\")|model_path = os.path.join(script_dir, \"lora_continue_training/continue_training_high_rank_direct\")|g" kaggle_submission.py

echo -e "\n===== 使用新模型生成提交文件 ====="
python kaggle_submission.py

# 验证提交文件
echo -e "\n===== 验证提交文件 ====="
python verify_submission.py "continue_training_submission.csv"

echo -e "\n===== 完成 ====="
echo "模型微调已完成，并生成了新的提交文件: $SCRIPT_DIR/continue_training_submission.csv"
echo "请将此文件上传到Kaggle竞赛页面。" 