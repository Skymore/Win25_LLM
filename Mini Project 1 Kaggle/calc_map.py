import pandas as pd
import numpy as np

def calculate_map(ground_truth_file, submission_file):
    """
    计算MAP@10分数
    """
    # 读取文件
    ground_truth_df = pd.read_csv(ground_truth_file)
    submission_df = pd.read_csv(submission_file)
    
    # 确保查询顺序一致
    assert all(ground_truth_df['Query'] == submission_df['Query']), "Queries don't match!"
    
    ap_scores = []
    
    for i in range(len(ground_truth_df)):
        # 获取ground truth和提交的文档ID
        gt_docs = ground_truth_df.iloc[i]['Doc_ID'].split()
        sub_docs = submission_df.iloc[i]['Doc_ID'].split()
        
        # 确保每个查询都有10个文档
        assert len(sub_docs) == 10, f"Query {i} doesn't have 10 documents!"
        
        # 计算这个查询的AP@10
        relevant_count = 0
        ap = 0.0
        
        for k, doc_id in enumerate(sub_docs, 1):
            if doc_id in gt_docs:
                relevant_count += 1
                precision_at_k = relevant_count / k
                ap += precision_at_k
        
        if relevant_count > 0:
            ap /= min(len(gt_docs), 10)  # normalize by min(relevant docs, 10)
        ap_scores.append(ap)
    
    # 计算MAP
    map_score = np.mean(ap_scores)
    
    print(f"MAP@10 Score: {map_score:.4f}")
    return map_score

# 计算MAP分数
map_score = calculate_map(
    'submission_from_beir_ground_truth_openai_large.csv',
    'submission_direct_openai_large.csv'
)