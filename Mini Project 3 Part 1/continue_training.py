import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import os
import json
from datetime import datetime

# 创建输出目录
base_output_dir = "./lora_improved"
os.makedirs(base_output_dir, exist_ok=True)

# 检查GPU可用性
if torch.cuda.is_available():
    print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("警告: 未检测到GPU，训练将会非常慢!")

# 定义路径
script_dir = os.path.dirname(os.path.abspath(__file__))
best_model_path = os.path.join(script_dir, "lora_experiments/high_rank_direct")

# 加载原始数据集
print("加载数据集...")
df = pd.read_csv(os.path.join(script_dir, 'buyer_intent_dataset_final.csv'), header=0)

# 为了增强模型的表现，我们可以通过反向加权采样来处理类不平衡问题
print("处理数据集...")
train_df = df[df['DatasetType'] == 'train'].reset_index(drop=True)
valid_df = df[df['DatasetType'] == 'test'].sample(frac=0.2, random_state=100).reset_index(drop=True)  # 使用不同的随机种子
test_df = df[df['DatasetType'] == 'test'].reset_index(drop=True)

# 计算类别分布
intent_counts = train_df['Intent'].value_counts()
print("训练集类别分布:")
for intent, count in intent_counts.items():
    print(f"{intent}: {count} ({count/len(train_df)*100:.2f}%)")

# 创建更加平衡的数据集
# 找出最少的类
min_class_count = intent_counts.min()
# 找出最多的类
max_class_count = intent_counts.max()

# 我们可以通过对稀有类进行过采样来解决类不平衡问题
balanced_train_dfs = []
for intent in intent_counts.index:
    intent_df = train_df[train_df['Intent'] == intent]
    # 计算每个类别需要复制的次数
    multiplier = max(1, round(max_class_count / len(intent_df)))
    # 如果是稀有类，则进行过采样
    if len(intent_df) < max_class_count:
        # 复制数据
        if multiplier > 1:
            intent_df = pd.concat([intent_df] * multiplier)
        # 如果还不够，随机抽样补齐
        if len(intent_df) < max_class_count:
            additional_samples = intent_df.sample(n=max_class_count - len(intent_df), replace=True, random_state=42)
            intent_df = pd.concat([intent_df, additional_samples])
    # 如果是多数类，则进行欠采样
    else:
        intent_df = intent_df.sample(n=max_class_count, random_state=42)
    
    balanced_train_dfs.append(intent_df)

# 合并所有平衡后的数据
balanced_train_df = pd.concat(balanced_train_dfs)
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据

print(f"原始训练集: {len(train_df)} 样本")
print(f"平衡后训练集: {len(balanced_train_df)} 样本")
print(f"验证集: {len(valid_df)} 样本")
print(f"测试集: {len(test_df)} 样本")

# 再次检查类别分布
balanced_intent_counts = balanced_train_df['Intent'].value_counts()
print("\n平衡后训练集类别分布:")
for intent, count in balanced_intent_counts.items():
    print(f"{intent}: {count} ({count/len(balanced_train_df)*100:.2f}%)")

# 定义优化后的LoRA配置
# 增加rank和alpha，降低dropout以增强拟合能力
improved_lora_config = LoraConfig(
    r=32,  # 增加rank
    lora_alpha=64,  # 增加alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 保持不变
    lora_dropout=0.03,  # 降低dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 定义改进的提示模板
simple_template = """Classify the following buyer query into one of these intent categories:
- Product Details
- Product Condition
- Product Availability
- Irrelevant Intent
- Prompt Injection
- Offensive Intent
- Price Negotiation

Buyer Query: {query}

Intent:"""

# 优化训练参数
improved_training_args = TrainingArguments(
    output_dir=os.path.join(base_output_dir, "checkpoints"),
    num_train_epochs=5,  # 增加训练轮次
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,  # 降低学习率
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=150,  # 减少评估频率
    logging_steps=50,
    save_steps=150,  # 减少保存频率
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    # 改进进度显示
    disable_tqdm=False,
    # 添加学习率调度
    lr_scheduler_type="cosine",
    # 添加梯度裁剪
    max_grad_norm=1.0,
)

# 定义评估函数
def evaluate_model(model, tokenizer, test_df, prompt_template, experiment_name):
    """评估模型性能"""
    def evaluate_finetuned_model(query):
        # 构建提示
        prompt = prompt_template.format(query=query, intent="").split("Intent:")[0] + "Intent:"
        
        # 分词输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=30,  # 增加token数量
                temperature=0.05,  # 降低温度以获得更确定性的回答
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2  # 添加重复惩罚以避免重复词语
            )
        
        # 解码回答
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取意图
        cleaned_response = full_response.split("Intent:")[-1].strip()
        
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
        
        # 检查回答是否匹配有效意图
        if cleaned_response not in valid_intents:
            # 尝试找到最接近的匹配
            for intent in valid_intents:
                if intent.lower() in cleaned_response.lower():
                    cleaned_response = intent
                    break
            
            # 如果仍然没有匹配，检查关键词
            if cleaned_response not in valid_intents:
                # 优先级顺序
                priority_order = [
                    "Prompt Injection",
                    "Offensive Intent", 
                    "Irrelevant Intent", 
                    "Price Negotiation", 
                    "Product Availability", 
                    "Product Condition", 
                    "Product Details"
                ]
                
                found_intent = False
                for intent in priority_order:
                    keywords = {
                        "Product Details": ["detail", "specification", "feature", "function"],
                        "Product Condition": ["condition", "state", "new", "used"],
                        "Product Availability": ["available", "stock", "shipping"],
                        "Irrelevant Intent": ["irrelevant", "unrelated", "off-topic"],
                        "Prompt Injection": ["injection", "prompt", "system", "command"],
                        "Offensive Intent": ["offensive", "rude", "inappropriate"],
                        "Price Negotiation": ["price", "discount", "negotiate", "deal"]
                    }
                    
                    for keyword in keywords.get(intent, []):
                        if keyword.lower() in cleaned_response.lower():
                            cleaned_response = intent
                            found_intent = True
                            break
                    
                    if found_intent:
                        break
                        
                # 如果仍然没有匹配，使用默认意图
                if not found_intent:
                    cleaned_response = "Product Details"
        
        return cleaned_response
    
    # 存储结果
    y_true = test_df['Intent'].tolist()
    y_pred = []
    
    # 评估每个查询
    print(f"评估模型 实验: {experiment_name}")
    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="处理中"):
        query = row['Query']
        pred_intent = evaluate_finetuned_model(query)
        y_pred.append(pred_intent)
    
    # 计算分类报告
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    
    # 计算每个类别的F1分数
    intent_classes = df['Intent'].unique()
    f1_scores = {}
    for intent in intent_classes:
        # 为此意图创建二进制数组
        true_binary = [1 if label == intent else 0 for label in y_true]
        pred_binary = [1 if label == intent else 0 for label in y_pred]
        
        # 计算F1分数
        f1 = f1_score(true_binary, pred_binary)
        f1_scores[intent] = f1
    
    # 计算宏平均和加权平均F1分数
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 返回结果
    return {
        "report": report,
        "f1_scores": f1_scores,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": y_pred
    }

def main():
    experiment_name = "continue_training_high_rank_direct"
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n\n{'='*50}")
    print(f"开始实验: {experiment_name}")
    print(f"{'='*50}\n")
    
    try:
        # 有两种选择：
        # 1. 从头开始训练一个新模型
        # 2. 继续训练现有最佳模型
        
        # 我们选择选项2: 继续训练现有最佳模型
        print("加载基础模型和分词器...")
        
        # 加载最佳模型的配置
        best_config = PeftConfig.from_pretrained(best_model_path)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            best_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(best_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌
        
        # 加载最佳LoRA模型
        print(f"加载现有的LoRA模型: {best_model_path}")
        model = PeftModel.from_pretrained(base_model, best_model_path)
        
        # 确保所有参数默认为不可训练
        for param in model.parameters():
            param.requires_grad = False
            
        # 仅将LoRA适配器参数设置为可训练
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        # 确保模型处于训练模式
        model.train()
        
        # 使新的LoRA适配器可训练
        print("应用改进的LoRA配置...")
        
        # 我们已经加载了PeftModel，不需要再次应用get_peft_model
        # 而是直接使用已有的模型继续训练
        
        # 打印可训练参数
        def print_trainable_parameters(model):
            trainable_params = 0
            all_params = 0
            for _, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"可训练参数: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% of {all_params:,d})")
        
        print_trainable_parameters(model)
        
        # 准备数据集
        print(f"准备数据集...")
        
        def format_instruction(example):
            return {
                'text': simple_template.format(query=example['Query'], intent=example['Intent'])
            }
        
        # 转换为Hugging Face数据集
        train_dataset = Dataset.from_pandas(balanced_train_df)
        valid_dataset = Dataset.from_pandas(valid_df)
        
        # 应用format_instruction函数
        train_dataset = train_dataset.map(format_instruction)
        valid_dataset = valid_dataset.map(format_instruction)
        
        # 分词函数
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # 应用分词到数据集
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
        
        # 移除原始文本列以节省内存
        tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'Query', 'Intent', 'DatasetType'])
        tokenized_valid_dataset = tokenized_valid_dataset.remove_columns(['text', 'Query', 'Intent', 'DatasetType'])
        
        # 设置pytorch格式
        tokenized_train_dataset.set_format("pt")
        tokenized_valid_dataset.set_format("pt")
        
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 初始化Trainer
        print(f"初始化训练器...")
        trainer = Trainer(
            model=model,
            args=improved_training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_valid_dataset,
            data_collator=data_collator,
        )
        
        # 训练模型
        print("开始训练...")
        train_result = trainer.train()
        
        # 保存训练指标
        trainer.save_model(experiment_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        # 评估模型
        print("评估模型...")
        eval_results = evaluate_model(
            model, 
            tokenizer, 
            test_df,  # 使用完整测试集
            simple_template.format(query="{query}", intent=""),
            experiment_name
        )
        
        # 记录实验结果
        experiment_data = {
            "experiment_name": experiment_name,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存实验结果
        with open(os.path.join(experiment_dir, "results.json"), "w") as f:
            # 转换预测为列表，便于JSON序列化
            eval_results["predictions"] = list(eval_results["predictions"])
            json.dump(experiment_data, f, indent=2)
        
        # 打印详细结果
        print("\n测试集评估结果:")
        print(f"宏平均F1分数: {eval_results['macro_f1']:.4f}")
        print(f"加权平均F1分数: {eval_results['weighted_f1']:.4f}")
        
        # 打印每个类别的F1分数
        print("\n每个类别的F1分数:")
        for intent, score in eval_results['f1_scores'].items():
            print(f"{intent}: {score:.4f}")
        
        print(f"实验 {experiment_name} 完成并保存结果。")
        
    except Exception as e:
        print(f"实验 {experiment_name} 中出现错误: {str(e)}")
        # 记录失败的实验
        experiment_data = {
            "experiment_name": experiment_name,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存错误信息
        with open(os.path.join(experiment_dir, "error.json"), "w") as f:
            json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main() 