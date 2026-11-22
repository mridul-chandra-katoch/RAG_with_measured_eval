import json
import random
from pathlib import Path
from ragkit.index import TFIDFIndex, BM25Index
from ragkit.retrieve import HybridRetriever, rrf
from ragkit.learned.features import split_articles, build_chunks
from ragkit.evals import mrr

SQUAD_DEFAULT = "datasets/squad/train-v2.0.json"

def cluster_bootstrap(a, b, groups, n=10000, seed=0):
    rng = random.Random(seed)
    by = {}
    for x, y, g in zip(a, b, groups):
        by.setdefault(g, []).append(x - y)
    keys = list(by)
    tot = {g: (sum(v), len(v)) for g, v in by.items()}
    out = []
    for _ in range(n):
        s = c = 0
        for _ in keys:
            gs, gc = tot[keys[rng.randrange(len(keys))]]
            s += gs; c += gc
        out.append(s / c)
    out.sort()
    return sum(x - y for x, y in zip(a, b)) / len(a), out[int(0.025 * n)], out[int(0.975 * n)]

def _load_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_index(articles, squad_data, top_k):
    chunks = build_chunks(articles, squad_data)
    tfidf = TFIDFIndex()
    tfidf.build(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    return HybridRetriever(tfidf, bm25, top_k)

def main():
    squad = _load_squad(SQUAD_DEFAULT)
    split = split_articles(len(squad["data"]), 150, 42)
    dev_articles = split["calibration"]
    
    retriever = _build_index(dev_articles, squad, top_k=10)
    
    bm25_rr = []
    tfidf_rr = []
    rrf_rr = []
    groups = []
    
    for art_idx in dev_articles:
        article = squad["data"][art_idx]
        for para_idx, para in enumerate(article["paragraphs"]):
            gold_id = f"a{art_idx}_p{para_idx}"
            for qa in para["qas"]:
                if qa["is_impossible"]: continue
                query = qa["question"]
                
                tfidf_res = retriever.tfidf.search(query, 10)
                bm25_res = retriever.bm25.search(query, 10)
                fused = rrf(tfidf_res, bm25_res)[:10]
                
                bm25_rr.append(mrr([c for c, _ in bm25_res], [gold_id]))
                tfidf_rr.append(mrr([c for c, _ in tfidf_res], [gold_id]))
                rrf_rr.append(mrr([c for c, _ in fused], [gold_id]))
                groups.append(art_idx)
                
    print(f"Dev Set Evaluation (n={len(bm25_rr)} queries)")
    print(f"BM25-only MRR:  {sum(bm25_rr)/len(bm25_rr):.4f}")
    print(f"TF-IDF-only MRR: {sum(tfidf_rr)/len(tfidf_rr):.4f}")
    print(f"RRF MRR:        {sum(rrf_rr)/len(rrf_rr):.4f}")
    
    mean_diff, ci_low, ci_high = cluster_bootstrap(rrf_rr, bm25_rr, groups)
    print(f"\nCluster Bootstrap (RRF - BM25):")
    print(f"Mean Diff: {mean_diff:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if ci_low <= 0 <= ci_high:
        print("Result: No detectable difference between RRF and BM25-only.")
    elif mean_diff > 0:
        print("Result: RRF significantly beats BM25-only.")
    else:
        print("Result: RRF is significantly worse than BM25-only.")

if __name__ == "__main__":
    main()