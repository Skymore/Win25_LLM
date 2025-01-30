## Sentense Embedder

1. **Vocabulary Setup**  
   We assume our vocabulary has 5 words:
   ```
   "i", "lot", "love", "chocolate", "milk"
   ```
   Each is assigned a 2D embedding based on its index \(i\), using the rule 
   \(\text{embedding}(i) = [\,i-1,\; i+1\,]\) for **0-indexed** \(i\).  
   Thus:
   - `"i"` (index 0)        \(\to [-1,  1]\)
   - `"lot"` (index 1)      \(\to [ 0,  2]\)
   - `"love"` (index 2)     \(\to [ 1,  3]\)
   - `"chocolate"` (index 3)\(\to [ 2,  4]\)
   - `"milk"` (index 4)     \(\to [ 3,  5]\)

   Any word **not** in this vocabulary defaults to \([-1, -1]\).

2. **Filler Words**  
   Filler words are `["i", "and", "as", "for", "it", "or", "maybe"]`.  
   Under the *skip-filler* method, if a token is in this list, we simply ignore it when averaging.

3. **Sentence**  
   We consider the (example) sentence  
   ```
   "I love chocolate milk as well!"
   ```
   For approach #3, we split `"well!"` into two tokens `"well"` and `"!"`, so we have 7 tokens:
   ```
   ["I", "love", "chocolate", "milk", "as", "well", "!"]
   ```
   to match the 7 example weights: `[0.5, 1.0, 0.7, 0.9, 0.3, 0.2, 0.1]`.

4. **Method 1: Simple Average Embedding**  
   - Include every token’s embedding.  
   - Vocabulary words get their defined embedding; out-of-vocab words get `[-1, -1]`.  
   - Then we just average across all tokens.

   In the sentence `"I love chocolate milk as well!"` (6 tokens if we keep “well!” together):
   ```
   I          => [-1,  1]  (in vocab)
   love       => [ 1,  3]  (in vocab)
   chocolate  => [ 2,  4]  (in vocab)
   milk       => [ 3,  5]  (in vocab)
   as         => [-1, -1]  (filler, but we do NOT skip in method #1; not in vocab => [-1,-1])
   well!      => [-1, -1]  (not in vocab)
   ```
   Summation = `[3, 11]`.  
   Number of tokens = 6.  
   Average = \(\bigl[\frac{3}{6}, \frac{11}{6}\bigr] \approx [0.50, 1.8333]\).

5. **Method 2: Skip Filler Words Embedding**  
   - We skip any filler word (e.g. `"I"` and `"as"`).  
   - We then average the rest.

   In the same sentence, skipping `"I"` and `"as"` leaves us:
   ```
   love       => [ 1,  3]
   chocolate  => [ 2,  4]
   milk       => [ 3,  5]
   well!      => [-1, -1]
   ```
   Summation = `[5, 11]`.  
   Number of tokens considered = 4.  
   Average = \(\bigl[\frac{5}{4}, \frac{11}{4}\bigr] = [1.25, 2.75]\).

6. **Method 3: Learned Sentence Embedding**  
   - We have a learned weight for each token, e.g. from a self-attention layer.  
   - We compute a **weighted average** of the token embeddings.

   Splitting the sentence as 7 tokens:  
   ```
   Tokens:        I    love  chocolate  milk   as    well    !
   Weights:       0.5  1.0   0.7        0.9    0.3   0.2     0.1
   Embeddings:   [-1,1],[1,3],[2,4],   [3,5],[-1,-1],[-1,-1],[-1,-1]
   ```
   Weighted sum =  
   \[
     0.5 \cdot [-1,1] \;+\; 
     1.0 \cdot [1,3]  \;+\; 
     0.7 \cdot [2,4]  \;+\; 
     0.9 \cdot [3,5]  \;+\; 
     0.3 \cdot [-1,-1]\;+\;
     0.2 \cdot [-1,-1]\;+\;
     0.1 \cdot [-1,-1].
   \]
   This sums to `[4.0, 10.2]` and the sum of weights is `3.7`.  
   Dividing gives:  
   \[
     \bigl[\tfrac{4.0}{3.7},\;\tfrac{10.2}{3.7}\bigr] \approx [1.0811,\;2.7568].
   \]

---

### Final Numeric Results

- **Simple Average**: \([0.5,\;1.8333]\)  
- **Skip Filler Average**: \([1.25,\;2.75]\)  
- **Learned (Weighted) Embedding**: \(\approx [1.08,\;2.76]\)
