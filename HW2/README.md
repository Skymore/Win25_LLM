



# Mini Project 1 - Text Similarity Search Documentation

## Team Information
```python
Team: [Team Number/Name]

Members:
1. [Name 1]
   - Implemented core algorithms (cosine_similarity, averaged_glove_embeddings_gdrive, get_sorted_cosine_similarity)
   - Conducted basic testing

2. [Name 2]
   - Deployed web application on Hugging Face Spaces
   - Created documentation and test cases
```

## 1. Implementation Overview

### 1.1 Core Functions (100 points)
```python
# Task I: Cosine Similarity (20 points)
def cosine_similarity(x, y):
    """
    Calculates exponentiated cosine similarity between two vectors
    """
    [Implementation details]

# Task II: GloVe Embeddings (30 points)
def averaged_glove_embeddings_gdrive(sentence, word_index_dict, embeddings, model_type=50):
    """
    Calculates averaged GloVe embeddings for input sentence
    """
    [Implementation details]

# Task III: Similarity Sorting (50 points)
def get_sorted_cosine_similarity(embeddings_metadata):
    """
    Calculates and sorts similarities between input and categories
    """
    [Implementation details]
```

## 2. Test Results

### 2.1 GloVe Models Test
```python
Categories: "Flowers Colors Cars Weather Food"
Test Query: "Roses are red, trucks are blue, and Seattle is grey right now"

# Results with different dimensions:
1. GloVe 25d
   [Screenshot 1]

2. GloVe 50d
   [Screenshot 2]

3. GloVe 100d
   [Screenshot 3]
```

### 2.2 Sentence Transformer Test
```python
# Same categories and query
Model: all-MiniLM-L6-v2
[Screenshot 4]
```

## 3. Web Application

### 3.1 Deployment (Bonus 10%)
- **URL**: [Your-Hugging-Face-Spaces-URL]
- **Platform**: Hugging Face Spaces

### 3.2 Usage Guide
1. Access the application using the provided URL
2. Enter categories in the text input field (space-separated)
   - Example: "Flowers Colors Cars Weather Food"
3. Enter your search query
   - Example: "Roses are red, trucks are blue"
4. Select embedding model dimension (25d/50d/100d)
5. View results in the pie chart visualization

## 4. Dependencies
```python
# requirements.txt
streamlit
numpy
sentence-transformers
matplotlib
gdown
```

## 5. Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run Mini_Project_1_Part_1.py
```

[Insert actual screenshots and results here]

需要我详细展开任何部分吗？或者需要调整文档结构？
