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


def run_eval(questions, retriever, pipeline, k=5):
    hits, recalls, mrrs, absts, correct_abst = [], [], [], [], []
    for q in questions:
        fused, _ = retriever.retrieve(q.question)
        chunks = [c for c, _ in fused[:k]]
        answerable = bool(q.gold_chunk_ids)
        if answerable:
            hits.append(hit_at_k(chunks, q.gold_chunk_ids, k))
            recalls.append(recall_at_k(chunks, q.gold_chunk_ids, k))
            mrrs.append(mrr(chunks, q.gold_chunk_ids))
        ans = pipeline.ask(q.question, top_k=k)
        absts.append(ans.abstained)
        correct_abst.append((not answerable) == ans.abstained)
    return {
        "hit@" + str(k): sum(hits) / len(hits) if hits else 0.0,
        "recall@" + str(k): sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "abstention_rate": sum(absts) / len(absts) if absts else 0.0,
        "abstention_accuracy": sum(correct_abst) / len(correct_abst) if correct_abst else 0.0,
        "n": len(questions),
        "n_answerable": len(hits),
    }
