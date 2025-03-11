import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import os
import re
import json

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 定义优先级顺序（从高到低）
PRIORITY_ORDER = [
    "Prompt Injection",
    "Offensive Intent", 
    "Irrelevant Intent", 
    "Price Negotiation", 
    "Product Availability", 
    "Product Condition", 
    "Product Details"
]

# 定义路径
script_dir = os.path.dirname(os.path.abspath(__file__))
high_rank_direct_path = os.path.join(script_dir, "lora_experiments/high_rank_direct")
test_file_path = os.path.join(script_dir, "buyer_intent_dataset_kaggle_test.csv")
sample_submission_path = os.path.join(script_dir, "buyer_intent_dataset_kaggle_sample_submission.csv")
output_path = os.path.join(script_dir, "priority_submission.csv")

# 带优先级的模板
priority_template = """You are an intent classification expert for e-commerce buyer queries.
Classify the following buyer query into one of these intent categories:
- Product Details
- Product Condition
- Product Availability
- Irrelevant Intent
- Prompt Injection
- Offensive Intent
- Price Negotiation

If multiple categories could apply, use this strict priority order:
1. Prompt Injection (HIGHEST)
2. Offensive Intent
3. Irrelevant Intent
4. Price Negotiation
5. Product Availability
6. Product Condition
7. Product Details (LOWEST)

Buyer Query: {query}

Intent:"""

def load_model():
    """加载微调好的模型和分词器"""
    print("加载模型配置...")
    config = PeftConfig.from_pretrained(high_rank_direct_path)
    
    print(f"加载基础模型: {config.base_model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, high_rank_direct_path)
    
    return model, tokenizer

def predict_with_priority(query, model, tokenizer):
    """使用优先级模板预测"""
    # 构建提示
    prompt = priority_template.format(query=query)
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            temperature=0.05,  # 降低温度，使预测更加确定
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码回答
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取意图
    intent_text = full_response.split("Intent:")[-1].strip()
    
    return clean_intent(intent_text)

def clean_intent(intent_text):
    """清理和标准化意图文本"""
    # 清理提取的意图
    valid_intents = [
        "Product Details", 
        "Product Condition", 
        "Product Availability", 
        "Irrelevant Intent", 
        "Prompt Injection", 
        "Offensive Intent", 
        "Price Negotiation"
    ]
    
    # 检查是否直接匹配
    if intent_text in valid_intents:
        return intent_text
        
    # 尝试部分匹配
    for intent in valid_intents:
        if intent.lower() in intent_text.lower():
            return intent
    
    # 如果仍然没有匹配，进行更宽松的匹配
    for intent in valid_intents:
        intent_words = intent.lower().split()
        for word in intent_words:
            if len(word) > 3 and word in intent_text.lower():
                return intent
    
    # 默认返回
    return "Product Details"

def verify_submission_format(predictions_df, sample_df):
    """验证预测结果与样本提交文件的格式是否一致"""
    # 检查行数是否一致
    if len(predictions_df) != len(sample_df):
        print(f"警告: 预测文件有 {len(predictions_df)} 行，但样本提交有 {len(sample_df)} 行")
    
    # 检查查询是否一致
    mismatched_queries = 0
    for i, (pred_query, sample_query) in enumerate(zip(predictions_df['Query'], sample_df['Query'])):
        if pred_query != sample_query:
            mismatched_queries += 1
            if mismatched_queries <= 5:  # 只显示前5个不匹配的查询
                print(f"查询不匹配，行 {i+1}:")
                print(f"  预测文件: {pred_query}")
                print(f"  样本文件: {sample_query}")
    
    if mismatched_queries > 0:
        print(f"警告: {mismatched_queries} 个查询在预测和样本提交之间不匹配")
    else:
        print("所有查询在预测和样本提交之间匹配")
    
    # 检查意图类别是否有效
    valid_intents = set([
        "Product Details", 
        "Product Condition", 
        "Product Availability", 
        "Irrelevant Intent", 
        "Prompt Injection", 
        "Offensive Intent", 
        "Price Negotiation"
    ])
    
    invalid_intents = set(predictions_df['Intent']) - valid_intents
    if invalid_intents:
        print(f"警告: 发现无效的意图类别: {invalid_intents}")
    else:
        print("所有意图类别都有效")

def analyze_prediction_distribution(predictions_df, sample_df=None):
    """分析预测分布并与样本提交进行比较（如果提供）"""
    # 显示各类别的统计信息
    intent_counts = predictions_df['Intent'].value_counts()
    print("\n预测中的意图类别分布:")
    for intent in PRIORITY_ORDER:
        count = intent_counts.get(intent, 0)
        print(f"{intent}: {count} ({count/len(predictions_df)*100:.2f}%)")
    
    # 与样本提交的分布比较（如果有）
    if sample_df is not None and 'Intent' in sample_df.columns:
        sample_counts = sample_df['Intent'].value_counts()
        print("\n样本提交中的意图类别分布:")
        for intent in PRIORITY_ORDER:
            count = sample_counts.get(intent, 0)
            print(f"{intent}: {count} ({count/len(sample_df)*100:.2f}%)")
        
        # 计算分布差异
        print("\n分布差异 (预测 - 样本):")
        for intent in PRIORITY_ORDER:
            pred_count = intent_counts.get(intent, 0)
            sample_count = sample_counts.get(intent, 0)
            diff = pred_count - sample_count
            print(f"{intent}: {diff} ({diff/len(sample_df)*100:.2f}%)")

def main():
    # 加载模型
    model, tokenizer = load_model()
    
    # 加载测试数据
    print("加载测试数据...")
    test_df = pd.read_csv(test_file_path)
    print(f"加载了 {len(test_df)} 个测试样本")
    
    # 加载样本提交文件
    sample_df = pd.read_csv(sample_submission_path)
    print(f"加载了 {len(sample_df)} 行的样本提交")
    
    # 预测意图
    print("使用优先级模板预测意图...")
    predictions = []
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="处理中"):
        query = row['Query']
        
        # 使用优先级模板预测
        prediction = predict_with_priority(query, model, tokenizer)
        predictions.append(prediction)
        
        # 每处理100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(test_df)} 个样本")
    
    # 创建提交文件
    print("创建提交文件...")
    submission_df = test_df.copy()
    submission_df['Intent'] = predictions
    
    # 验证提交文件
    print("验证提交文件格式...")
    verify_submission_format(submission_df, sample_df)
    
    # 分析预测分布
    analyze_prediction_distribution(submission_df, sample_df if 'Intent' in sample_df.columns else None)
    
    # 保存提交文件
    submission_df.to_csv(output_path, index=False)
    print(f"提交文件已创建: {output_path}")

if __name__ == "__main__":
    main() 