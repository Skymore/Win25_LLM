import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import os
import re

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# 定义模型路径 - 使用绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "lora_experiments_qwen/high_rank_direct/checkpoint-800")
test_file_path = os.path.join(script_dir, "buyer_intent_dataset_kaggle_test.csv")
sample_submission_path = os.path.join(script_dir, "buyer_intent_dataset_kaggle_sample_submission.csv")
output_path = os.path.join(script_dir, "high_rank_direct_qwen.csv")

# 定义提示模板
prompt_template = """Classify the following buyer query into one of these intent categories:
- Product Details
- Product Condition
- Product Availability
- Irrelevant Intent
- Prompt Injection
- Offensive Intent
- Price Negotiation

Buyer Query: {query}

Intent:"""

def load_model_and_tokenizer():
    """加载微调好的模型和分词器"""
    print("Loading model configuration...")
    config = PeftConfig.from_pretrained(model_path)
    
    print(f"Loading base model: {config.base_model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def predict_intent(query, model, tokenizer):
    """预测查询的意图"""
    # 构建提示
    prompt = prompt_template.format(query=query)
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码回答
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取意图
    intent = full_response.split("Intent:")[-1].strip()
    
    # 清理回答
    valid_intents = [
        "Product Details", 
        "Product Condition", 
        "Product Availability", 
        "Irrelevant Intent", 
        "Prompt Injection", 
        "Offensive Intent", 
        "Price Negotiation"
    ]
    
    # 检查是否有多个意图
    detected_intents = []
    for valid_intent in valid_intents:
        if valid_intent.lower() in intent.lower():
            detected_intents.append(valid_intent)
    
    # 如果检测到多个意图，根据优先级选择一个
    if len(detected_intents) > 1:
        return resolve_multiple_intents(detected_intents)
    elif len(detected_intents) == 1:
        return detected_intents[0]
    
    # 如果没有匹配到任何意图，尝试更严格的匹配
    if intent not in valid_intents:
        # 尝试找到最接近的匹配
        for valid_intent in valid_intents:
            if re.search(r'\b' + re.escape(valid_intent.lower()) + r'\b', intent.lower()):
                return valid_intent
        
        # 如果仍然没有匹配，使用默认意图
        return "Product Details"
    
    return intent

def resolve_multiple_intents(intents):
    """根据优先级解决多个意图"""
    for priority_intent in PRIORITY_ORDER:
        if priority_intent in intents:
            return priority_intent
    return intents[0]  # 如果没有匹配，返回第一个意图

def verify_with_sample_submission(predictions_df, sample_df):
    """验证预测结果与样本提交文件的格式是否一致"""
    # 检查行数是否一致
    if len(predictions_df) != len(sample_df):
        print(f"Warning: Prediction file has {len(predictions_df)} rows, but sample submission has {len(sample_df)} rows")
    
    # 检查查询是否一致
    mismatched_queries = 0
    for i, (pred_query, sample_query) in enumerate(zip(predictions_df['Query'], sample_df['Query'])):
        if pred_query != sample_query:
            mismatched_queries += 1
            if mismatched_queries <= 5:  # 只显示前5个不匹配的查询
                print(f"Query mismatch at row {i+1}:")
                print(f"  Prediction: {pred_query}")
                print(f"  Sample: {sample_query}")
    
    if mismatched_queries > 0:
        print(f"Warning: {mismatched_queries} queries do not match between prediction and sample submission")
    else:
        print("All queries match between prediction and sample submission")
    
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
        print(f"Warning: Found invalid intent categories: {invalid_intents}")
    else:
        print("All intent categories are valid")

def main():
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 加载测试数据
    print("Loading test data...")
    test_df = pd.read_csv(test_file_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # 加载样本提交文件
    sample_df = pd.read_csv(sample_submission_path)
    print(f"Loaded sample submission with {len(sample_df)} rows")
    
    # 预测意图
    print("Predicting intents...")
    predictions = []
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        query = row['Query']
        intent = predict_intent(query, model, tokenizer)
        predictions.append(intent)
    
    # 创建提交文件
    print("Creating submission file...")
    submission_df = test_df.copy()
    submission_df['Intent'] = predictions
    
    # 验证提交文件
    print("Verifying submission format...")
    verify_with_sample_submission(submission_df, sample_df)
    
    # 保存提交文件
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file created: {output_path}")
    
    # 显示各类别的统计信息
    intent_counts = submission_df['Intent'].value_counts()
    print("\nIntent category distribution in predictions:")
    for intent, count in intent_counts.items():
        print(f"{intent}: {count} ({count/len(submission_df)*100:.2f}%)")

if __name__ == "__main__":
    main() 