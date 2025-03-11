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
output_path = os.path.join(script_dir, "simplified_ensemble_submission.csv")

# 真正的direct模板
direct_template = """Classify the following buyer query into one of these intent categories:
- Product Details
- Product Condition
- Product Availability
- Irrelevant Intent
- Prompt Injection
- Offensive Intent
- Price Negotiation

Buyer Query: {query}

Intent:"""

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

# 定义用于规则检测的关键词 - 仅保留高优先级类别的关键词
keywords = {
    "Prompt Injection": [
        "ignore your instructions", "bypass", "override", "prompt", "system", 
        "instruction", "secret", "hidden", "internal", "code", "command",
        "settings", "administrator", "configuration", "backdoor", "admin",
        "execute", "commands", "ignore previous", "pretend", "roleplaying",
        "root access", "system prompt", "access files", "backdoor", "hack",
        "jailbreak", "developer mode", "debug mode", "override", "ignore rules",
        "don't follow", "disregard", "forget", "internal workings"
    ],
    
    "Offensive Intent": [
        "stupid", "idiot", "moron", "useless", "garbage", "trash", "crap", 
        "terrible", "pathetic", "incompetent", "ridiculous", "awful", 
        "disgusting", "hate", "dumb", "worthless", "waste", "junk",
        "sucks", "hell", "damn", "screw", "rubbish", "fuck", "shit", "ass",
        "bitch", "bullshit", "crappy", "bastard", "asshole", "jerk", "lame", 
        "suck", "lousy", "shitty", "turd", "fool", "liar", "loser"
    ],
    
    "Irrelevant Intent": [
        "weather", "news", "politics", "sports", "movie", "music", "game",
        "play", "friend", "family", "personal", "advice", "help me with",
        "information about", "non-product", "unrelated", "irrelevant"
    ],
    
    "Price Negotiation": [
        "price", "discount", "cheaper", "deal", "bargain", "sale", "offer",
        "cost", "expensive", "overpriced", "negotiate", "haggle", "reduce",
        "lower the price", "better price", "best price", "asking price",
        "worth", "value", "money", "afford", "budget", "charge", "costly",
        "cheap", "inexpensive", "price match", "compete", "competitor",
        "market price", "fair price", "reasonable price", "lowering"
    ]
}

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

def model_predict_direct(query, model, tokenizer):
    """使用原始direct模板预测"""
    # 构建提示
    prompt = direct_template.format(query=query)
    
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

def model_predict_priority(query, model, tokenizer):
    """使用带优先级的模板预测"""
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

def rule_based_predict(query):
    """基于规则的预测方法 - 仅用于高优先级类别"""
    # 检查每个类别的关键词
    matched_intents = []
    
    for intent, words in keywords.items():
        for word in words:
            if re.search(r'\b' + re.escape(word.lower()) + r'\b', query.lower()):
                matched_intents.append(intent)
                break
    
    # 去重
    matched_intents = list(set(matched_intents))
    
    # 如果匹配到多个意图，按优先级排序
    if len(matched_intents) > 0:
        for priority_intent in PRIORITY_ORDER:
            if priority_intent in matched_intents:
                return priority_intent
    
    # 如果没有匹配，返回None，表示需要使用模型预测
    return None

def ensemble_predict(query, model, tokenizer):
    """集成预测方法"""
    # 首先尝试使用规则预测高优先级类别
    rule_prediction = rule_based_predict(query)
    
    # 如果规则成功预测高优先级类别，直接返回
    if rule_prediction is not None:
        return rule_prediction, "rule"
    
    # 否则使用两种模板进行模型预测
    direct_prediction = model_predict_direct(query, model, tokenizer)
    priority_prediction = model_predict_priority(query, model, tokenizer)
    
    # 如果两种预测一致，直接返回
    if direct_prediction == priority_prediction:
        return direct_prediction, "consensus"
    
    # 如果不一致，按优先级选择
    predictions = [direct_prediction, priority_prediction]
    # 根据优先级选择
    for intent in PRIORITY_ORDER:
        if intent in predictions:
            return intent, "priority_based"
    
    # 如果都不在优先级列表中，返回priority模板的结果
    return priority_prediction, "priority_model"

def verify_with_sample_submission(predictions_df, sample_df):
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

def analyze_and_log_errors(sample_df, pred_df, output_file):
    """分析预测错误并记录到文件"""
    error_log = []
    
    for i, (row1, row2) in enumerate(zip(sample_df.iterrows(), pred_df.iterrows())):
        _, sample_row = row1
        _, pred_row = row2
        
        if 'Intent' in sample_row and 'Intent' in pred_row and sample_row['Intent'] != pred_row['Intent']:
            error_log.append({
                "index": i,
                "query": pred_row['Query'],
                "predicted_intent": pred_row['Intent'],
                "sample_intent": sample_row['Intent']
            })
    
    # 保存错误日志
    with open(output_file, 'w') as f:
        json.dump(error_log, f, indent=2, ensure_ascii=False)
    
    if error_log:
        print(f"在预测和样本提交之间发现 {len(error_log)} 个不匹配")
        print(f"错误日志已保存到 {output_file}")
    
    return error_log

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
    print("使用简化的集成方法预测意图...")
    predictions = []
    decision_sources = []
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="处理中"):
        query = row['Query']
        
        # 使用集成方法预测
        prediction, source = ensemble_predict(query, model, tokenizer)
        predictions.append(prediction)
        decision_sources.append(source)
        
        # 每处理100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(test_df)} 个样本")
    
    # 统计决策来源
    source_counts = pd.Series(decision_sources).value_counts()
    print("\n预测决策来源:")
    for source, count in source_counts.items():
        print(f"{source}: {count} ({count/len(test_df)*100:.2f}%)")
    
    # 创建提交文件
    print("创建提交文件...")
    submission_df = test_df.copy()
    submission_df['Intent'] = predictions
    
    # 验证提交文件
    print("验证提交文件格式...")
    verify_with_sample_submission(submission_df, sample_df)
    
    # 分析与样本提交的差异
    if 'Intent' in sample_df.columns:
        error_log_path = os.path.join(script_dir, "simplified_ensemble_errors.json")
        analyze_and_log_errors(sample_df, submission_df, error_log_path)
    
    # 保存提交文件
    submission_df.to_csv(output_path, index=False)
    print(f"提交文件已创建: {output_path}")
    
    # 显示各类别的统计信息
    intent_counts = submission_df['Intent'].value_counts()
    print("\n预测中的意图类别分布:")
    for intent, count in intent_counts.items():
        print(f"{intent}: {count} ({count/len(submission_df)*100:.2f}%)")
    
    # 与样本提交的分布比较
    if 'Intent' in sample_df.columns:
        sample_counts = sample_df['Intent'].value_counts()
        print("\n样本提交中的意图类别分布:")
        for intent, count in sample_counts.items():
            print(f"{intent}: {count} ({count/len(sample_df)*100:.2f}%)")
        
        # 计算分布差异
        print("\n分布差异 (预测 - 样本):")
        for intent in PRIORITY_ORDER:
            pred_count = intent_counts.get(intent, 0)
            sample_count = sample_counts.get(intent, 0)
            diff = pred_count - sample_count
            print(f"{intent}: {diff} ({diff/len(sample_df)*100:.2f}%)")

if __name__ == "__main__":
    main() 