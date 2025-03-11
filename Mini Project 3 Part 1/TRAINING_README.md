# 买家意图分类模型微调改进

本文档介绍了如何继续微调买家意图分类模型以提高Kaggle竞赛分数。

## 改进策略

为了提高模型性能，我们采用了以下改进策略：

1. **数据平衡**：处理训练集中的类别不平衡问题，通过过采样稀有类别和欠采样多数类别来创建更平衡的数据集。

2. **优化LoRA配置**：
   - 增加rank (r=32)和alpha (α=64)以增强模型容量
   - 降低dropout (0.03)以增强拟合能力

3. **改进提示模板**：
   - 添加更详细的类别描述
   - 明确指定类别优先级顺序
   - 强调优先级规则

4. **优化训练参数**：
   - 增加训练轮次 (5轮)
   - 降低学习率 (1e-5)
   - 添加学习率调度 (余弦调度)
   - 添加早停策略
   - 增加梯度裁剪

5. **推理优化**：
   - 降低温度以获得更确定性的回答
   - 增加生成的token数量
   - 添加重复惩罚
   - 改进意图提取和清理逻辑

## 使用方法

### 1. 继续微调模型

运行以下命令开始微调：

```bash
./run_continue_training.sh
```

该脚本会：
1. 运行`continue_training.py`继续微调现有的最佳模型
2. 自动修改预测脚本中的模型路径
3. 使用新模型生成Kaggle提交文件
4. 验证提交文件格式

### 2. 使用微调后的模型预测

如果只想使用已微调的模型进行预测，可以运行：

```bash
python improved_kaggle.py
```

### 3. 验证提交文件

可以使用以下命令验证生成的提交文件：

```bash
python verify_submission.py "improved_kaggle_submission.csv"
```

## 文件说明

- `continue_training.py`: 继续微调模型的主脚本
- `improved_kaggle.py`: 使用改进的预测方法生成提交文件
- `run_continue_training.sh`: 运行整个微调和预测流程的脚本
- `verify_submission.py`: 验证提交文件格式的脚本

## 模型路径

- 原始最佳模型: `lora_experiments/high_rank_direct`
- 改进后的模型: `lora_improved/improved_high_rank_direct`

## 改进效果

通过上述改进，我们预期可以：

1. 提高对稀有类别的识别能力
2. 更准确地处理多意图查询，按照优先级正确分类
3. 提高整体的宏平均F1分数
4. 特别提高在Prompt Injection和Offensive Intent类别上的表现 