#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import ndcg_score

from ragkit.index import TFIDFIndex, BM25Index, tokenize
from ragkit.retrieve import HybridRetriever, rrf
from ragkit.learned.features import split_articles, build_chunks
from ragkit.learned.reranker import extract_pair_features
from ragkit.evals import mrr

SQUAD_DEFAULT = "datasets/squad/train-v2.0.json"
RESULTS_JSON = Path("results/reranker_results.json")
REPORT_MD = Path("results/reranker_report.md")
MODEL_JSON = Path("models/reranker_xgb.json")

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

def _mine_rank_data(retriever, articles, squad_data, top_k=10, max_qs=5000):
    X, y, qid = [], [], []
    processed = 0
    for art_idx in articles:
        article = squad_data["data"][art_idx]
        for para_idx, para in enumerate(article["paragraphs"]):
            gold_id = _get_gold_id(art_idx, para_idx)
            for qa in para["qas"]:
                if qa["is_impossible"]: continue
                query = qa["question"]
                q_tokens = tokenize(query)
                
                tfidf_res = retriever.tfidf.search(query, top_k)
                bm25_res = retriever.bm25.search(query, top_k)
                fused = rrf(tfidf_res, bm25_res)
                
                tfidf_map = {c.chunk_id: s for c, s in tfidf_res}
                bm25_map = {c.chunk_id: s for c, s in bm25_res}
                tfidf_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(tfidf_res)}
                bm25_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(bm25_res)}
                
                top_bm25 = bm25_res[0][1] if bm25_res else 0.0
                top_tfidf = tfidf_res[0][1] if tfidf_res else 0.0
                
                has_gold = False
                for chunk, rrf_score in fused[:top_k]:
                    c_tokens = tokenize(chunk.text)
                    try: c_pos = int(chunk.chunk_id.split("_p")[1])
                    except: c_pos = 0
                    
                    feat = extract_pair_features(
                        q_tokens, query, c_tokens, chunk.text,
                        bm25_map.get(chunk.chunk_id, 0.0),
                        tfidf_map.get(chunk.chunk_id, 0.0),
                        rrf_score,
                        bm25_rank_map.get(chunk.chunk_id, 0),
                        tfidf_rank_map.get(chunk.chunk_id, 0),
                        c_pos, top_bm25, top_tfidf
                    )
                    X.append(feat)
                    y.append(1 if chunk.chunk_id == gold_id else 0)
                    qid.append(processed)
                    if chunk.chunk_id == gold_id: has_gold = True
                
                if has_gold:
                    processed += 1
                else:
                    # Remove the query if the gold chunk wasn't even retrieved
                    X = X[:-len(fused[:top_k])]
                    y = y[:-len(fused[:top_k])]
                    qid = qid[:-len(fused[:top_k])]
                    
                if processed >= max_qs:
                    return np.array(X), np.array(y), np.array(qid)
    return np.array(X), np.array(y), np.array(qid)

def _evaluate_mrr(retriever, articles, squad_data, model=None, top_k=10):
    rrf_mrrs, bm25_mrrs, rn_mrrs = [], [], []
    
    for art_idx in articles:
        article = squad_data["data"][art_idx]
        for para_idx, para in enumerate(article["paragraphs"]):
            gold_id = _get_gold_id(art_idx, para_idx)
            for qa in para["qas"]:
                if qa["is_impossible"]: continue
                query = qa["question"]
                q_tokens = tokenize(query)
                
                tfidf_res = retriever.tfidf.search(query, top_k)
                bm25_res = retriever.bm25.search(query, top_k)
                fused = rrf(tfidf_res, bm25_res)[:top_k]
                
                if not fused:
                    rrf_mrrs.append(0.0); bm25_mrrs.append(0.0); rn_mrrs.append(0.0)
                    continue
                    
                rrf_mrrs.append(mrr([c for c, _ in fused], [gold_id]))
                bm25_mrrs.append(mrr([c for c, _ in bm25_res], [gold_id]))
                
                if model is not None:
                    tfidf_map = {c.chunk_id: s for c, s in tfidf_res}
                    bm25_map = {c.chunk_id: s for c, s in bm25_res}
                    tfidf_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(tfidf_res)}
                    bm25_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(bm25_res)}
                    top_bm25 = bm25_res[0][1] if bm25_res else 0.0
                    top_tfidf = tfidf_res[0][1] if tfidf_res else 0.0
                    
                    X_test = []
                    for chunk, rrf_score in fused:
                        c_tokens = tokenize(chunk.text)
                        try: c_pos = int(chunk.chunk_id.split("_p")[1])
                        except: c_pos = 0
                        raw_f = extract_pair_features(
                            q_tokens, query, c_tokens, chunk.text,
                            bm25_map.get(chunk.chunk_id, 0.0),
                            tfidf_map.get(chunk.chunk_id, 0.0),
                            rrf_score,
                            bm25_rank_map.get(chunk.chunk_id, 0),
                            tfidf_rank_map.get(chunk.chunk_id, 0),
                            c_pos, top_bm25, top_tfidf
                        )
                        X_test.append(raw_f)
                    
                    scores = model.predict(np.array(X_test))
                    ranked = sorted(zip(fused, scores), key=lambda x: -x[1])
                    rn_mrrs.append(mrr([c for (c, _), _ in ranked], [gold_id]))
                else:
                    rn_mrrs.append(rrf_mrrs[-1])
                    
    return rrf_mrrs, bm25_mrrs, rn_mrrs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=150)
    parser.add_argument("--max-train-questions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    squad = _load_squad(SQUAD_DEFAULT)
    split = split_articles(len(squad["data"]), args.max_articles, args.seed)
    
    print("[index] building train index...")
    train_retriever = _build_index(split["train"], squad, args.top_k)
    
    print(f"[mining] extracting up to {args.max_train_questions} questions...")
    X_train, y_train, qid_train = _mine_rank_data(train_retriever, split["train"], squad, args.top_k, args.max_train_questions)
    print(f"[mining] extracted {len(X_train)} rows")

    print(f"[train] fitting XGBoost Ranker...")
    # XGBoost ranking model
    model = xgb.XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=args.seed,
        n_jobs=4
    )
    model.fit(X_train, y_train, qid=qid_train)

    print(f"[eval] evaluating on test partition...")
    test_retriever = _build_index(split["test"], squad, args.top_k)
    rrf_mrrs, bm25_mrrs, rn_mrrs = _evaluate_mrr(test_retriever, split["test"], squad, model, args.top_k)
    
    mean_rrf = sum(rrf_mrrs) / len(rrf_mrrs)
    mean_bm25 = sum(bm25_mrrs) / len(bm25_mrrs)
    mean_rn = sum(rn_mrrs) / len(rn_mrrs)
    
    print(f"[eval] RRF MRR: {mean_rrf:.4f}")
    print(f"[eval] BM25-only MRR: {mean_bm25:.4f}")
    print(f"[eval] XGBoost MRR: {mean_rn:.4f}")
    
    if mean_rn > mean_bm25:
        print("[eval] XGBoost beats BM25-only!")
    else:
        print("[eval] XGBoost does NOT beat BM25-only.")

    # Save model
    MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_JSON))

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps({
        "rrf_mrr": mean_rrf,
        "bm25_mrr": mean_bm25,
        "xgb_mrr": mean_rn,
    }, indent=2))

    print(f"Saved {RESULTS_JSON}, {REPORT_MD}, {MODEL_JSON}")

if __name__ == "__main__":
    main()