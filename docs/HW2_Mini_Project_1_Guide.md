# Mini Project 1: Information Retrieval with BEIR Dataset - 学习指南

## 项目概述 Project Overview

本项目旨在介绍学生使用BEIR数据集进行信息检索任务。主要学习目标包括：

1. 理解BEIR数据集的结构并进行数据预处理
2. 实现一个系统来对查询和文档进行编码
3. 计算相似度分数以对文档进行相关性排序
4. 使用平均精度均值(MAP)等指标评估系统性能
5. 修改和微调模型以获得更好的检索结果

## 详细任务说明

### Task 1: 编码查询和文档 (10分)

#### 实现目标
- 使用GloVe词向量对句子进行编码
- 通过平均每个句子中的单词向量来生成句子嵌入
- 处理未知词的情况

#### 关键实现
```python
def encode_with_glove(self, glove_file_path: str, sentences: list[str]) -> list[np.ndarray]:
    """
    输入:
    - glove_file_path: GloVe嵌入文件的路径
    - sentences: 需要编码的句子列表
    
    输出:
    - 句子嵌入的列表
    
    实现步骤:
    1. 加载GloVe词向量
    2. 对每个句子:
       - 分词
       - 查找每个词的向量
       - 对未知词使用零向量
       - 计算平均向量作为句子嵌入
    """
```

### Task 2: 计算余弦相似度和文档排序 (20分)

#### 实现目标
- 计算查询和文档之间的余弦相似度
- 基于相似度对文档进行排序
- 保存排序结果用于后续评估

#### 关键实现
```python
def rank_documents(self, encoding_method: str = 'sentence_transformer'):
    """
    输入:
    - encoding_method: 编码方法 ('glove' 或 'sentence_transformer')
    
    实现步骤:
    1. 根据选择的方法对查询和文档进行编码
    2. 计算查询和文档间的余弦相似度
    3. 对每个查询的文档进行排序
    4. 保存排序结果
    """
```

### Task 3: 评估系统性能 (10分)

#### 实现目标
- 实现平均精度(AP)计算，只考虑前k个文档（k=10）
- 计算所有查询的平均精度均值(MAP@10)
- 评估系统整体性能

#### 关键实现
```python
@staticmethod
def average_precision(relevant_docs: list[str], candidate_docs: list[str], k: int = 10) -> float:
    """
    实现步骤:
    1. 只取前k个候选文档
    2. 计算这k个文档中的相关文档数
    3. 计算MAP@k
    
    注意：
    - k通常设置为10，因为在实际应用中用户很少会查看更多结果
    - 这种方式更符合实际使用场景
    - 可以更好地评估模型在最相关文档上的表现
    """
    # 只取前k个文档
    candidate_docs = candidate_docs[:k]
    # 计算这k个文档中哪些是相关的
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
    # 计算每个位置的precision
    precisions = [np.mean(y_true[:i+1]) for i in range(len(y_true)) if y_true[i]]
    return np.mean(precisions) if precisions else 0
```

#### 为什么要限制为top-k？
1. **实际应用考虑**
   - 用户通常只关注搜索结果的前几个
   - 前k个结果的质量最重要
   - 符合实际使用场景

2. **评估重点**
   - 更关注模型将相关文档排在前面的能力
   - 减少后面不太相关文档的干扰
   - 提供更有意义的性能度量

3. **计算效率**
   - 减少计算量
   - 更快得到评估结果
   - 便于模型比较和优化

### Task 4: 基于相似度分数对文档排序 (10分)

#### 实现目标
- 展示给定查询的相关文档排序
- 显示相似度分数
- 输出前10个最相关的文档

#### 关键实现
```python
def show_ranking_documents(self, example_query: str):
    """
    实现步骤:
    1. 对输入查询进行编码
    2. 计算与所有文档的相似度
    3. 选择并展示前K个最相关文档
    4. 打印文档ID和相似度分数
    """
```

### Task 5: 微调Sentence Transformer模型 (25分)

#### 实现目标
- 使用MultipleNegativesRankingLoss进行模型微调
- 实验不同的训练策略
- 评估微调后的模型性能

#### 关键实现
```python
def fine_tune_model(self, batch_size: int = 32, num_epochs: int = 3):
    """
    实现步骤:
    1. 准备训练样本
    2. 设置损失函数
    3. 冻结部分模型层
    4. 训练模型
    5. 保存微调后的模型
    """
```

## 评分标准

### 报告部分 1: 文档排序 (10分)
- 分析不同编码方法的性能
- 比较GloVe和Sentence Transformer的效果
- 讨论余弦相似度的表现
- 提出可能的改进方案

### 报告部分 2: 模型微调 (15分)
- 比较不同训练策略的效果
- 分析MAP分数的变化
- 讨论训练损失和学习率的影响
- 提出未来改进方向

## 学习建议

1. **数据集理解**
   - 仔细阅读BEIR数据集的文档
   - 理解数据集的结构和格式
   - 熟悉查询-文档的关系

2. **编码方法**
   - 深入理解GloVe和Sentence Transformer的原理
   - 比较两种方法的优缺点
   - 注意处理边界情况（如未知词）

3. **相似度计算**
   - 理解余弦相似度的数学原理
   - 注意计算效率和内存使用
   - 考虑其他可能的相似度度量

4. **模型微调**
   - 理解MultipleNegativesRankingLoss的工作原理
   - 合理设置训练参数
   - 注意过拟合问题

5. **性能评估**
   - 理解MAP指标的含义
   - 正确实现评估指标
   - 分析性能瓶颈

## 常见问题

1. **为什么使用GloVe和Sentence Transformer两种方法？**
   - GloVe是静态词向量，简单但效果有限
   - Sentence Transformer考虑了上下文，效果更好
   - 对比两种方法有助于理解嵌入的重要性

2. **如何处理大规模数据集？**
   - 使用批处理进行编码
   - 优化内存使用
   - 考虑使用GPU加速

3. **如何改进检索效果？**
   - 尝试不同的预训练模型
   - 调整微调策略
   - 优化文本预处理

4. **为什么要计算MAP？**
   - MAP是评估排序系统的标准指标
   - 考虑了精确度和召回率
   - 能够全面评估系统性能

## 参考资源

1. BEIR数据集文档：
   - https://huggingface.co/datasets/BeIR/nfcorpus
   - https://huggingface.co/datasets/BeIR/nfcorpus-qrels

2. Sentence Transformers文档：
   - https://www.sbert.net/

3. 相关论文：
   - BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models
   - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

## 时间管理建议

1. **第一阶段 (2-3天)**
   - 理解项目需求
   - 熟悉数据集
   - 实现基础编码功能

2. **第二阶段 (2-3天)**
   - 实现文档排序
   - 计算相似度
   - 评估基础系统

3. **第三阶段 (3-4天)**
   - 实现模型微调
   - 优化系统性能
   - 进行实验对比

4. **最后阶段 (2-3天)**
   - 撰写报告
   - 总结发现
   - 提出改进建议 