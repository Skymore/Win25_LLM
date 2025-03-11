#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "切换到脚本目录: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

echo "运行最简单提示词+规则预测..."
python simple_rule_submission.py

echo "验证提交文件..."
python verify_submission.py "simple_rule_submission.csv"

echo "完成！"
echo "提交文件已生成: $SCRIPT_DIR/simple_rule_submission.csv"
echo "请将此文件上传到Kaggle竞赛页面。" 