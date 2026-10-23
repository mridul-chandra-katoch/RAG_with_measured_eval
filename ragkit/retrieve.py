def rrf(dense, lexical, k=60):
    scores = {}
    chunk_map = {}
    for rank, (c, _) in enumerate(dense, 1):
        scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[c.chunk_id] = c
    for rank, (c, _) in enumerate(lexical, 1):
        scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[c.chunk_id] = c
    return [(chunk_map[cid], s) for cid, s in sorted(scores.items(), key=lambda x: -x[1])]


class HybridRetriever:
    def __init__(self, tfidf, bm25, top_k=20):
        self.tfidf = tfidf
        self.bm25 = bm25
        self.top_k = top_k

    def retrieve(self, query):
        tfidf_results = self.tfidf.search(query, self.top_k)
        bm25_results = self.bm25.search(query, self.top_k)
        fused = rrf(tfidf_results, bm25_results)
        top_raw = tfidf_results[0][1] if tfidf_results else 0.0
        return fused, top_raw
