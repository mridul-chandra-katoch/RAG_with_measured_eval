import json

def hit_at_k(chunks, gold_ids, k):
    return 1.0 if any(c.chunk_id in gold_ids for c in chunks[:k]) else 0.0

def recall_at_k(chunks, gold_ids, k):
    if not gold_ids:
        return 1.0
    got = {c.chunk_id for c in chunks[:k]}
    return len(got & set(gold_ids)) / len(gold_ids)

def mrr(chunks, gold_ids):
    for r, c in enumerate(chunks, 1):
        if c.chunk_id in gold_ids:
            return 1.0 / r
    return 0.0

def run_benchmark(questions, retriever, k=5):
    hits, recalls, mrrs = [], [], []
    
    for q in questions:
        if not q.get("answerable", False):
            continue
            
        gold_span_norm = " ".join(str(q["gold_span"]).lower().split())
        gold_doc_clean = str(q["gold_doc"]).replace(".", "_")
        
        # Find matching chunks
        gold_chunk_ids = []
        for c in retriever.tfidf.chunks:
            c_text_norm = " ".join(c.text.lower().split())
            if gold_span_norm in c_text_norm:
                gold_chunk_ids.append(c.chunk_id)
                
        if not gold_chunk_ids:
            # Fallback to document matching if exact span isn't found
            gold_chunk_ids = [c.chunk_id for c in retriever.tfidf.chunks if gold_doc_clean in c.doc_id]
            
        fused, _ = retriever.retrieve(q["question"])
        chunks = [c for c, _ in fused[:k]]
        
        hits.append(hit_at_k(chunks, gold_chunk_ids, k))
        recalls.append(recall_at_k(chunks, gold_chunk_ids, k))
        mrrs.append(mrr(chunks, gold_chunk_ids))
        
    n = len(hits)
    return {
        f"hit@{k}": sum(hits)/n if n else 0.0,
        f"recall@{k}": sum(recalls)/n if n else 0.0,
        "mrr": sum(mrrs)/n if n else 0.0,
        "n_answerable": n,
        "n_total": len(questions)
    }