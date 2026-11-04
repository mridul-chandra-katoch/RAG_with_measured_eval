"""From-scratch pairwise learning-to-rank (RankNet-style) reranker.

Features are extracted for a (query, chunk) pair. The model is trained
on pairs (positive_chunk, negative_chunk) for the same query using a
pairwise logistic loss (RankNet).

No ragkit imports here: testable on its own.
"""

import math
import random

def sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

def extract_pair_features(query_tokens, chunk_tokens, bm25_score, tfidf_score, rrf_score):
    """Extract features for a single (query, chunk) pair."""
    q_set = set(query_tokens)
    c_set = set(chunk_tokens)
    
    overlap = len(q_set & c_set)
    jaccard = overlap / len(q_set | c_set) if (q_set | c_set) else 0.0
    
    # Exact match proportion: how many query words appear in the chunk
    q_match = overlap / len(q_set) if q_set else 0.0
    
    # Features:
    # 1. BM25 raw score
    # 2. TF-IDF cosine score
    # 3. RRF score
    # 4. Query length
    # 5. Chunk length
    # 6. Word overlap count
    # 7. Jaccard similarity
    # 8. Query match proportion
    
    return [
        bm25_score,
        tfidf_score,
        rrf_score,
        float(len(query_tokens)),
        float(len(chunk_tokens)),
        float(overlap),
        jaccard,
        q_match,
    ]

def train_ranknet(
    X_pairs, 
    l2=1e-3, 
    lr=0.01, 
    epochs=200, 
    batch_size=32, 
    seed=42
):
    """Train a pairwise RankNet."""
    n = len(X_pairs)
    if n == 0:
        raise ValueError("X_pairs is empty")
    
    d = len(X_pairs[0][0])
    w = [0.0] * d
    # Bias is not used in ranking, so we drop it entirely to prevent numerical instability
    rng = random.Random(seed)
    
    for epoch in range(1, epochs + 1):
        order = list(range(n))
        rng.shuffle(order)
        
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            g_w = [0.0] * d
            m = len(batch)
            
            for i in batch:
                x_pos, x_neg = X_pairs[i]
                
                s_pos = sum(wj * xj for wj, xj in zip(w, x_pos))
                s_neg = sum(wj * xj for wj, xj in zip(w, x_neg))
                s_diff = s_pos - s_neg
                
                # P(pos > neg) = sigmoid(s_diff)
                # Gradient of loss w.r.t s_diff is sigmoid(-s_diff)
                p = sigmoid(-s_diff)
                
                for j in range(d):
                    grad = p * (x_pos[j] - x_neg[j])
                    g_w[j] += grad
                
            # Update weights (ADD the gradient, apply L2 penalty)
            for j in range(d):
                w[j] += lr * (g_w[j] / m) - lr * l2 * w[j]
            
    return w, 0.0  # Return bias as 0.0

def score_chunks(w, b, X):
    """Score a list of feature vectors."""
    return [sum(wj * xj for wj, xj in zip(w, x)) + b for x in X]