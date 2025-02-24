from sentence_transformers import SentenceTransformer
import numpy as np
import time

class SemanticSimilarityCalculator:
    def __init__(self, model_name='all-mpnet-base-v2', device='cpu'):
        """
        Initialize the semantic similarity calculator with a pre-trained model.
        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
        """
        self.model = SentenceTransformer(model_name, device=device)
    
    def calculate_similarity_batch(self, texts: list) -> np.ndarray:
        """
        Compute pairwise semantic similarity for a list of texts.
        Args:
            texts (list): List of company names (strings).
        Returns:
            np.ndarray: Pairwise similarity matrix (size: len(texts) x len(texts)).
        """
        # Encode all texts in a batch
        embeddings = self.model.encode(texts, normalize_embeddings=True)  # Normalize for cosine similarity

        # Compute cosine similarity using matrix multiplication
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix

def get_company_names():
    """Generate a list of 100 well-known company names, including multiple variations of the same company."""
    return [
        # 1. 公司全称 vs. 简称
        "Tesla", "Tesla Inc.",
        "Microsoft", "Microsoft Corporation",
        "Amazon", "Amazon.com Inc.",
        "Google", "Google LLC",
        "Facebook", "Meta",
        "YouTube", "Google",
        "Instagram", "Meta",
        "WhatsApp", "Meta",
        "LinkedIn", "Microsoft",
        "Windows", "Microsoft",
        
        # 2. 缩写 vs. 全称
        "IBM", "International Business Machines",
        "GE", "General Electric",
        "P&G", "Procter & Gamble",
        "3M", "Minnesota Mining and Manufacturing",
        "HP", "Hewlett-Packard",
        
        # 3. 拼写变体
        "Netflix", "NetFlix Inc.",
        "eBay", "Ebay.com",
        "PayPal", "Paypal Holdings",
        "Uber", "Uber Technologies",
        "Lyft", "Lyft Inc.",
        
        # 4. 品牌 vs. 旗下产品
        "Apple", "Apple Inc.",
        "iPhone", "Apple",
        "MacBook", "Apple",
        "iPad", "Apple",
        
        # 5. 相关品牌
        "PlayStation", "Sony",
        "Xperia", "Sony",
        "Samsung", "Samsung Electronics",
        "Galaxy", "Samsung",
        
        # 6. 主要服务
        "AWS", "Amazon Web Services",
        "Azure", "Microsoft Cloud",
        "Google Cloud", "Google",
        "Alibaba Cloud", "Alibaba",
        
        # 7. 母公司 vs. 子公司
        "Toyota", "Lexus",
        "Volkswagen", "Audi",
        "Volkswagen", "Porsche",
        "GM", "Chevrolet",
        "GM", "Cadillac",
        
        # 8. 银行与金融
        "Visa", "Visa Inc.",
        "Mastercard", "Mastercard International",
        "American Express", "Amex",
        "JPMorgan", "JPMorgan Chase",
        "Goldman Sachs", "Goldman Sachs Group",
        
        # 9. 医药
        "Pfizer", "Pfizer Inc.",
        "Moderna", "Moderna Therapeutics",
        "Johnson & Johnson", "J&J",
        "Novartis", "Novartis AG",
        "AstraZeneca", "AstraZeneca PLC",
        
        # 10. 航空航天
        "SpaceX", "Space Exploration Technologies",
        "Boeing", "The Boeing Company",
        "Lockheed Martin", "Lockheed",
        "Northrop Grumman", "Grumman",
        "Blue Origin", "Amazon Space",
        
        # 11. 传统工业
        "Shell", "Royal Dutch Shell",
        "ExxonMobil", "Exxon Mobil Corporation",
        "BP", "British Petroleum",
        "Chevron", "Chevron Corporation",
        "TotalEnergies", "Total S.A.",
        
        # 12. 其他品牌
        "Reddit", "Reddit Inc.",
        "Snapchat", "Snap Inc.",
        "TikTok", "ByteDance",
        "WeChat", "Tencent",
        "Alibaba", "AliExpress",
        "Tencent", "Tencent Holdings"
    ]

def main():
    # Initialize calculator
    calculator = SemanticSimilarityCalculator(device="cpu")  # Change to "cuda" if using GPU
    
    # Get predefined company names
    company_names = get_company_names()
    company_names.extend(company_names)

    time_start = time.time()

    # Compute similarity matrix
    similarity_matrix = calculator.calculate_similarity_batch(company_names)

    time_end = time.time()
    
    print(f"Total company names: {len(company_names)}")

    print(f"Computed similarity for {len(company_names)} company names.")
    print(f"Time taken: {time_end - time_start:.4f} seconds")

    # Print pairs with similarity > 0.8
    threshold = 0.8
    print("\nCompany name pairs with similarity above 0.8:")
    
    printed_pairs = set()
    
    for i in range(len(company_names)):
        for j in range(i + 1, len(company_names)):  # Avoid duplicate pairs
            similarity_score = similarity_matrix[i, j]
            if similarity_score > threshold:
                pair = (company_names[i], company_names[j])
                if pair not in printed_pairs:
                    print(f"{pair[0]:<30} ↔ {pair[1]:<30} Similarity: {similarity_score:.4f}")
                    printed_pairs.add(pair)

if __name__ == "__main__":
    main()
