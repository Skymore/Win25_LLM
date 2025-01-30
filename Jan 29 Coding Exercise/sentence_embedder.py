class SentenceEmbedder:
    def __init__(self):
        """
        Initialize vocabulary embeddings, filler words, and (optionally) 
        a fixed set of attention weights for demonstration.
        """
        # 1) Vocabulary and their 2D embeddings (i is 0-indexed):
        #    "i" -> index 0  => [-1, 1]
        #    "lot" -> index 1 -> [0, 2]
        #    "love" -> index 2 -> [1, 3]
        #    "chocolate" -> index 3 -> [2, 4]
        #    "milk" -> index 4 -> [3, 5]
        self.vocab_embeddings = {
            "i":          [-1,  1],
            "lot":        [ 0,  2],
            "love":       [ 1,  3],
            "chocolate":  [ 2,  4],
            "milk":       [ 3,  5]
        }
        
        # 2) Embedding for words not in vocabulary
        self.unk_embedding = [-1, -1]
        
        # 3) List of filler words
        self.filler_words = {"i", "and", "as", "for", "it", "or", "maybe"}
        
        # 4) Example learned attention weights for each token in the sentence
        #    "I love chocolate milk as well !"
        #    This example has 7 tokens (splitting 'well' and '!')
        self.learned_weights_example = [0.5, 1.0, 0.7, 0.9, 0.3, 0.2, 0.1]
    
    def get_word_embedding(self, word):
        """
        Return the 2D embedding for a single word if it is in the vocabulary;
        otherwise, return the 'unknown' embedding [-1, -1].
        """
        word_lower = word.lower().strip("!.?,")  # strip punctuation for lookup
        if word_lower in self.vocab_embeddings:
            return self.vocab_embeddings[word_lower]
        else:
            return self.unk_embedding
    
    def simple_average_embedding(self, sentence):
        """
        1) Simple average embedding.
        
        For each word in the sentence:
         - If the word is in the vocabulary, use its embedding.
         - If not in the vocabulary, use [-1, -1].
        Then average over all tokens in the sentence.
        """
        tokens = sentence.split()
        sum_emb = [0.0, 0.0]
        count = 0
        
        for token in tokens:
            emb = self.get_word_embedding(token)
            sum_emb[0] += emb[0]
            sum_emb[1] += emb[1]
            count += 1
        
        # Average
        if count > 0:
            return [sum_emb[0] / count, sum_emb[1] / count]
        else:
            return [0.0, 0.0]
    
    def skip_filler_embedding(self, sentence):
        """
        2) Skip filler words embedding.
        
        We skip any token that appears in the filler word list 
        and then average the rest.
        """
        tokens = sentence.split()
        sum_emb = [0.0, 0.0]
        count = 0
        
        for token in tokens:
            # Check if the "lowercased" token (minus punctuation) is in filler words
            token_lower = token.lower().strip("!.?,")
            if token_lower in self.filler_words:
                # Skip filler
                continue
            
            # Otherwise, get its embedding and add to sum
            emb = self.get_word_embedding(token)
            sum_emb[0] += emb[0]
            sum_emb[1] += emb[1]
            count += 1
        
        # Average over the non-filler tokens
        if count > 0:
            return [sum_emb[0] / count, sum_emb[1] / count]
        else:
            return [0.0, 0.0]
    
    def learned_sentence_embedding(self, sentence, weights=None):
        """
        3) Learned sentence embedding.
        
        We assume we have a pre-learned weight for each token 
        (for example, from a self-attention layer). We compute 
        a weighted average of the token embeddings, dividing by 
        the sum of the weights.
        
        - `weights`: if provided, must match the number of tokens in `sentence`.
        - Otherwise, we use `self.learned_weights_example` as a demonstration.
        """
        tokens = sentence.split()
        
        if weights is None:
            # For demo, we use the 7-element example, 
            # but your real code might handle varying lengths
            weights = self.learned_weights_example
        
        if len(tokens) != len(weights):
            raise ValueError("Number of tokens and number of weights must match!")
        
        weighted_sum = [0.0, 0.0]
        total_weight = 0.0
        
        for token, w in zip(tokens, weights):
            emb = self.get_word_embedding(token)
            weighted_sum[0] += emb[0] * w
            weighted_sum[1] += emb[1] * w
            total_weight += w
        
        if total_weight > 0:
            return [weighted_sum[0] / total_weight, weighted_sum[1] / total_weight]
        else:
            return [0.0, 0.0]

# ---------------------------
# Demonstration:
if __name__ == "__main__":
    embedder = SentenceEmbedder()
    
    sentence = "I love chocolate milk as well!"
    
    # 1) Simple average embedding
    emb_simple = embedder.simple_average_embedding(sentence)
    
    # 2) Skip filler words embedding
    emb_skip_filler = embedder.skip_filler_embedding(sentence)
    
    # 3) Learned sentence embedding (splitting 'well!' into 'well' and '!')
    #    So let's rewrite the sentence carefully to match 7 tokens:
    sentence_tokens = ["I", "love", "chocolate", "milk", "as", "well", "!"]
    sentence_7 = " ".join(sentence_tokens)
    emb_learned = embedder.learned_sentence_embedding(sentence_7)
    
    print("1) Simple average embedding:", emb_simple)
    print("2) Skip filler words embedding:", emb_skip_filler)
    print("3) Learned sentence embedding:", emb_learned)
    
    # OUTPOT:
    # (llm596) (interview) ➜  Win25_LLM /home/sky/miniforge3/envs/llm596/bin/python "/home/sky/projects/Win25_LLM/Jan 29 Coding Exercise/sentence_embedder.py"
    # 1) Simple average embedding: [0.5, 1.8333333333333333]
    # 2) Skip filler words embedding: [1.25, 2.75]
    # 3) Learned sentence embedding: [1.081081081081081, 2.756756756756757]
