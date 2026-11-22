"""Feature extraction for the learned reranker."""

def extract_pair_features(query_tokens, query_text, chunk_tokens, chunk_text, 
                          bm25_score, tfidf_score, rrf_score, 
                          bm25_rank, tfidf_rank, chunk_position,
                          bm25_top_score, tfidf_top_score):
    q_set = set(query_tokens)
    c_set = set(chunk_tokens)
    
    overlap = len(q_set & c_set)
    term_coverage = overlap / len(q_set) if q_set else 0.0
    
    q_norm = " ".join(query_text.lower().split())
    c_norm = " ".join(chunk_text.lower().split())
    exact_match = 1.0 if q_norm in c_norm else 0.0
    
    jaccard = overlap / len(q_set | c_set) if (q_set | c_set) else 0.0
    
    bm25_margin = bm25_score - bm25_top_score
    tfidf_margin = tfidf_score - tfidf_top_score
    
    return [
        bm25_score, tfidf_score, rrf_score,
        bm25_rank, tfidf_rank, exact_match,
        term_coverage, jaccard, float(len(chunk_tokens)),
        float(chunk_position), bm25_margin, tfidf_margin
    ]