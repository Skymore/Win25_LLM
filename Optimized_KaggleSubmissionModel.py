import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


class KaggleSubmissionModel:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the model with the pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.load_data()
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by converting it to lower-case, removing punctuation, and extra spaces.
        """
        # Convert to lower case
        text = text.lower()
        # Remove punctuation using regex
        text = re.sub(r"[^\w\s]", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_data(self):
        """
        Load test queries and documents from CSV files and preprocess them.
        """
        # Load test query and document CSV files
        self.test_queries_df = pd.read_csv('test_query.csv')
        self.test_documents_df = pd.read_csv('test_documents.csv')
        
        # Preprocess queries and documents
        self.queries = [self.preprocess_text(q) for q in self.test_queries_df['Query'].tolist()]
        self.documents = [self.preprocess_text(doc) for doc in self.test_documents_df['Doc'].tolist()]
        
        # Create document IDs using index as ID
        self.document_ids = [f'MED-{i}' for i in range(len(self.documents))]
        
        print(f'Loaded {len(self.queries)} queries and {len(self.documents)} documents')

    def rank_documents(self, batch_size: int = 32):
        """
        Rank documents for each query using sentence transformer embeddings.
        """
        print('Encoding queries...')
        query_embeddings = self.model.encode(self.queries, batch_size=batch_size, show_progress_bar=True)
        
        print('Encoding documents...')
        doc_embeddings = self.model.encode(self.documents, batch_size=batch_size, show_progress_bar=True)
        
        print('Computing similarities and ranking documents...')
        results = []
        
        # For each query compute cosine similarity with document embeddings
        for i, query_embedding in enumerate(tqdm(query_embeddings, desc='Ranking queries')):
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            # Get top 10 document indices based on similarity scores
            top_indices = np.argsort(similarities)[::-1][:10]
            top_doc_ids = [self.document_ids[idx] for idx in top_indices]
            doc_ids_str = ' '.join(top_doc_ids)
            results.append({
                'Query': self.test_queries_df.iloc[i]['Query'],
                'Doc_ID': doc_ids_str
            })
        
        # Save results to submission file
        submission_df = pd.DataFrame(results)
        submission_df.to_csv('submission.csv', index=False)
        print('Submission file created successfully!')
        
        # Display first few rows as example
        print('\nFirst few rows of the submission file:')
        print(submission_df.head())


def main():
    print('Initializing model with preprocessing optimizations...')
    model = KaggleSubmissionModel()
    print('Ranking documents...')
    model.rank_documents()
    

if __name__ == '__main__':
    main() 