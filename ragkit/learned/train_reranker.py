#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from pathlib import Path

from ragkit.index import TFIDFIndex, BM25Index, tokenize
from ragkit.retrieve import HybridRetriever, rrf
from ragkit.learned.features import split_articles, build_chunks
from ragkit.learned.reranker import extract_pair_features, train_ranknet, score_chunks
from ragkit.evals import mrr

SQUAD_DEFAULT = "datasets/squad/train-v2.0.json"
RESULTS_JSON = Path("results/reranker_results.json")
REPORT_MD = Path("results/reranker_report.md")

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

def _get_gold_id(article_idx, para_idx):
    return f"a{article_idx}_p{para_idx}"

def _mine_pairs(retriever, articles, squad_data, top_k=10, max_qs=2000):
    X_pairs = []
    processed = 0
    
    for art_idx in articles:
        article = squad_data["data"][art_idx]
        for para_idx, para in enumerate(article["paragraphs"]):
            gold_id = _get_gold_id(art_idx, para_idx)
            for qa in para["qas"]:
                if qa["is_impossible"]:
                    continue
                query = qa["question"]
                q_tokens = tokenize(query)
                
                tfidf_res = retriever.tfidf.search(query, top_k)
                bm25_res = retriever.bm25.search(query, top_k)
                fused = rrf(tfidf_res, bm25_res)
                
                tfidf_map = {c.chunk_id: s for c, s in tfidf_res}
                bm25_map = {c.chunk_id: s for c, s in bm25_res}
                
                pos_feat = None
                neg_feats = []
                
                for chunk, rrf_score in fused[:top_k]:
                    c_tokens = tokenize(chunk.text)
                    feat = extract_pair_features(
                        q_tokens, c_tokens,
                        bm25_map.get(chunk.chunk_id, 0.0),
                        tfidf_map.get(chunk.chunk_id, 0.0),
                        rrf_score
                    )
                    if chunk.chunk_id == gold_id:
                        pos_feat = feat
                    else:
                        neg_feats.append(feat)
                
                if pos_feat is not None and neg_feats:
                    for neg in neg_feats:
                        X_pairs.append((pos_feat, neg))
                        
                processed += 1
                if processed % 500 == 0:
                    print(f"[mining] processed {processed} questions, found {len(X_pairs)} pairs...")
                if processed >= max_qs:
                    print(f"[mining] reached max_qs ({max_qs}) limit.")
                    return X_pairs
                        
    return X_pairs

def _compute_standardization(X_pairs):
    all_feats = [f for pair in X_pairs for f in pair]
    n = len(all_feats)
    d = len(all_feats[0])
    
    means = [sum(f[j] for f in all_feats) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((f[j] - means[j]) ** 2 for f in all_feats) / n
        std = math.sqrt(var)
        stds.append(std if std > 0 else 1.0)
    return means, stds

def _standardize_single(feat, means, stds):
    return [(feat[j] - means[j]) / stds[j] for j in range(len(feat))]

def _evaluate_mrr(retriever, articles, squad_data, means, stds, w=None, b=None, top_k=10):
    rrf_mrrs = []
    rn_mrrs = []
    art_ids = []
    
    for art_idx in articles:
        article = squad_data["data"][art_idx]
        for para_idx, para in enumerate(article["paragraphs"]):
            gold_id = _get_gold_id(art_idx, para_idx)
            for qa in para["qas"]:
                if qa["is_impossible"]:
                    continue
                query = qa["question"]
                q_tokens = tokenize(query)
                
                tfidf_res = retriever.tfidf.search(query, top_k)
                bm25_res = retriever.bm25.search(query, top_k)
                fused = rrf(tfidf_res, bm25_res)[:top_k]
                
                if not fused:
                    rrf_mrrs.append(0.0)
                    rn_mrrs.append(0.0)
                    art_ids.append(art_idx)
                    continue
                    
                rrf_chunks = [c for c, _ in fused]
                rrf_mrrs.append(mrr(rrf_chunks, [gold_id]))
                
                if w is not None and b is not None:
                    tfidf_map = {c.chunk_id: s for c, s in tfidf_res}
                    bm25_map = {c.chunk_id: s for c, s in bm25_res}
                    
                    feats = []
                    for chunk, rrf_score in fused:
                        c_tokens = tokenize(chunk.text)
                        raw_f = extract_pair_features(
                            q_tokens, c_tokens,
                            bm25_map.get(chunk.chunk_id, 0.0),
                            tfidf_map.get(chunk.chunk_id, 0.0),
                            rrf_score
                        )
                        feats.append(_standardize_single(raw_f, means, stds))
                        
                    scores = score_chunks(w, b, feats)
                    ranked = sorted(zip(fused, scores), key=lambda x: -x[1])
                    rn_chunks = [c for (c, _), _ in ranked]
                    rn_mrrs.append(mrr(rn_chunks, [gold_id]))
                else:
                    rn_mrrs.append(rrf_mrrs[-1])
                    
                art_ids.append(art_idx)
                
    return rrf_mrrs, rn_mrrs, art_ids

def _bootstrap_mrr_diff(mrr_a, mrr_b, article_ids, n_resamples=1000, seed=42):
    rng = random.Random(seed)
    unique = sorted(set(article_ids))
    groups = {}
    for i, a in enumerate(article_ids):
        groups.setdefault(a, []).append(i)
        
    diffs = []
    for _ in range(n_resamples):
        picks = [rng.choice(unique) for _ in range(len(unique))]
        idx = []
        for a in picks:
            idx.extend(groups[a])
        m_a = sum(mrr_a[i] for i in idx) / len(idx)
        m_b = sum(mrr_b[i] for i in idx) / len(idx)
        diffs.append(m_a - m_b)
        
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    return lo, hi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=150)
    parser.add_argument("--max-train-questions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    squad = _load_squad(SQUAD_DEFAULT)
    split = split_articles(len(squad["data"]), args.max_articles, args.seed)
    train_articles = split["train"]
    test_articles = split["test"]

    print(f"[index] building train index...")
    train_retriever = _build_index(train_articles, squad, args.top_k)
    
    print(f"[mining] extracting up to {args.max_train_questions} questions...")
    t0 = time.time()
    X_pairs = _mine_pairs(train_retriever, train_articles, squad, args.top_k, args.max_train_questions)
    print(f"[mining] extracted {len(X_pairs)} pairs in {time.time()-t0:.1f}s")

    means, stds = _compute_standardization(X_pairs)
    X_pairs_std = [
        (_standardize_single(pos, means, stds), _standardize_single(neg, means, stds))
        for pos, neg in X_pairs
    ]

    print(f"[train] fitting RankNet...")
    t0 = time.time()
    w, b = train_ranknet(X_pairs_std, epochs=args.epochs, lr=args.lr, seed=args.seed)
    print(f"[train] done in {time.time()-t0:.1f}s")
    print(f"[train] weights={[round(x, 4) for x in w]}")

    print(f"[eval] evaluating on test partition...")
    test_retriever = _build_index(test_articles, squad, args.top_k)
    rrf_mrrs, rn_mrrs, art_ids = _evaluate_mrr(
        test_retriever, test_articles, squad, means, stds, w, b, args.top_k
    )
    
    mean_rrf = sum(rrf_mrrs) / len(rrf_mrrs)
    mean_rn = sum(rn_mrrs) / len(rn_mrrs)
    
    print(f"[eval] RRF MRR: {mean_rrf:.4f}")
    print(f"[eval] RankNet MRR: {mean_rn:.4f}")
    
    lo, hi = _bootstrap_mrr_diff(rn_mrrs, rrf_mrrs, art_ids, seed=args.seed)
    print(f"[eval] MRR diff CI: [{lo:.4f}, {hi:.4f}]")
    excludes_zero = (lo > 0) or (hi < 0)
    print(f"[eval] CI excludes 0: {excludes_zero}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "config": vars(args),
            "rrf_mrr": mean_rrf,
            "ranknet_mrr": mean_rn,
            "mrr_diff_ci": [lo, hi],
            "excludes_zero": excludes_zero,
            "weights": w,
            "means": means,
            "stds": stds
        }, f, indent=2)

    with open(REPORT_MD, "w") as f:
        f.write("# Learned Reranker Evaluation\n\n")
        f.write(f"- **Baseline (RRF) MRR**: {mean_rrf:.4f}\n")
        f.write(f"- **RankNet MRR**: {mean_rn:.4f}\n")
        f.write(f"- **MRR Difference (RankNet - RRF)**: {mean_rn - mean_rrf:.4f}\n")
        f.write(f"- **95% Bootstrap CI**: [{lo:.4f}, {hi:.4f}]\n")
        f.write(f"- **CI excludes 0**: {excludes_zero}\n\n")
        f.write("## Learned Weights\n")
        f.write(f"Weights: {[round(x, 4) for x in w]}\n")

    print(f"Saved {RESULTS_JSON} and {REPORT_MD}")

if __name__ == "__main__":
    main()