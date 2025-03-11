import pandas as pd
import os
import sys

def verify_submission(prediction_file, sample_file):
    """验证提交文件与样本提交文件的格式是否一致"""
    # 加载文件
    try:
        predictions_df = pd.read_csv(prediction_file)
        sample_df = pd.read_csv(sample_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        return False
    
    # 检查列名
    if set(predictions_df.columns) != set(sample_df.columns):
        print(f"Column mismatch: Prediction has {predictions_df.columns.tolist()}, Sample has {sample_df.columns.tolist()}")
        return False
    
    # 检查行数是否一致
    if len(predictions_df) != len(sample_df):
        print(f"Row count mismatch: Prediction has {len(predictions_df)} rows, Sample has {len(sample_df)} rows")
        return False
    
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
        return False
    
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
        return False
    
    # 显示各类别的统计信息
    intent_counts = predictions_df['Intent'].value_counts()
    print("\nIntent category distribution in predictions:")
    for intent, count in intent_counts.items():
        print(f"{intent}: {count} ({count/len(predictions_df)*100:.2f}%)")
    
    # 比较与样本提交的分布差异
    sample_counts = sample_df['Intent'].value_counts()
    print("\nIntent category distribution in sample submission:")
    for intent, count in sample_counts.items():
        print(f"{intent}: {count} ({count/len(sample_df)*100:.2f}%)")
    
    print("\nVerification passed! The submission file is valid.")
    return True

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 默认文件路径
    prediction_file = os.path.join(script_dir, "kaggle_submission.csv")
    sample_file = os.path.join(script_dir, "buyer_intent_dataset_kaggle_sample_submission.csv")
    
    # 允许从命令行指定文件
    if len(sys.argv) > 1:
        prediction_file = sys.argv[1]
    if len(sys.argv) > 2:
        sample_file = sys.argv[2]
    
    print(f"Verifying submission file: {prediction_file}")
    print(f"Against sample file: {sample_file}")
    
    verify_submission(prediction_file, sample_file)

if __name__ == "__main__":
    main() 