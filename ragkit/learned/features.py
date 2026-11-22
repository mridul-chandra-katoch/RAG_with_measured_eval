"""Feature extraction for the learned abstention classifier."""
import json
import random
import time
from pathlib import Path
from ragkit.index import BM25Index, TFIDFIndex, tokenize
from ragkit.retrieve import HybridRetriever, rrf
from ragkit.schemas import Chunk

FEATURE_COLUMNS = ["f1_bm25", "f2_bm25_gap", "f3_tfidf", "f4_tfidf_gap", "f5a_rank_match", "f5b_jaccard", "f6_query_len", "f7_token_match", "f8_retriever_gap"]
FEATURE_GROUP_COLUMNS = {
    "f1": [0], "f2": [1], "f3": [2], "f4": [3],
    "f5": [4, 5], "f6": [6], "f7": [7], "f8": [8]
}

def split_articles(num_articles, max_articles, seed):
    if max_articles < 10:
        raise ValueError(f"max_articles must be >= 10, got {max_articles}")
    if max_articles > num_articles:
        raise ValueError(f"max_articles ({max_articles}) > num_articles ({num_articles})")
    
    rng = random.Random(seed)
    sampled = rng.sample(range(num_articles), max_articles)
    n_train = round(0.60 * max_articles)
    n_cal = round(0.15 * max_articles)
    train = sorted(sampled[:n_train])
    cal = sorted(sampled[n_train:n_train + n_cal])
    test = sorted(sampled[n_train + n_cal:])
    return {"train": train, "calibration": cal, "test": test}

def assign_indexed_withheld(article_ids, seed, partition_name, indexed_fraction=0.7):
    rng = random.Random(f"{seed}:{partition_name}")
    shuffled = list(article_ids)
    rng.shuffle(shuffled)
    n_indexed = max(1, min(len(shuffled) - 1, round(indexed_fraction * len(shuffled))))
    return sorted(shuffled[:n_indexed]), sorted(shuffled[n_indexed:])

def build_chunks(article_indices, squad_data):
    chunks = []
    for art_idx in article_indices:
        article = squad_data["data"][art_idx]
        title = article.get("title", "")
        for para_idx, para in enumerate(article["paragraphs"]):
            chunks.append(Chunk(
                chunk_id=f"a{art_idx}_p{para_idx}",
                doc_id=f"a{art_idx}",
                text=para["context"],
                source={"file": title, "page": para_idx},
            ))
    return chunks

def extract_features(query_text, retriever):
    tfidf_raw = retriever.tfidf.search(query_text, retriever.top_k)
    bm25_raw = retriever.bm25.search(query_text, retriever.top_k)
    
    # Apply zero-score filtering (preserves items with score > 0.0)
    tfidf_results = [(c, s) for c, s in tfidf_raw if s > 0.0]
    bm25_results = [(c, s) for c, s in bm25_raw if s > 0.0]
    
    f1 = bm25_results[0][1] if bm25_results else 0.0
    f2 = (bm25_results[0][1] - bm25_results[1][1]) if len(bm25_results) >= 2 else (bm25_results[0][1] if bm25_results else 0.0)
    f3 = tfidf_results[0][1] if tfidf_results else 0.0
    f4 = (tfidf_results[0][1] - tfidf_results[1][1]) if len(tfidf_results) >= 2 else (tfidf_results[0][1] if tfidf_results else 0.0)
    
    f5a = 0.0
    if bm25_results and tfidf_results:
        f5a = 1.0 if bm25_results[0][0].chunk_id == tfidf_results[0][0].chunk_id else 0.0
        
    bm25_top5 = {c.chunk_id for c, _ in bm25_results[:5]}
    tfidf_top5 = {c.chunk_id for c, _ in tfidf_results[:5]}
    union = bm25_top5 | tfidf_top5
    f5b = (len(bm25_top5 & tfidf_top5) / len(union)) if union else 0.0

    qtokens = tokenize(query_text)
    f6 = float(len(qtokens))
    
    f7 = 0.0
    if bm25_results and qtokens:
        unique_q = set(qtokens)
        top1_tokens = set(tokenize(bm25_results[0][0].text))
        f7 = len(unique_q & top1_tokens) / len(unique_q)
        
    f8 = f1 - f3 # BM25 vs TF-IDF gap

    return [f1, f2, f3, f4, f5a, f5b, f6, f7, f8]

def build_or_load_cache(squad_data, split_map, indexed_map, withheld_map, top_k, seed, max_articles, cache_path, rebuild=False, log_every=500):
    index_sizes = {n: sum(len(squad_data["data"][i]["paragraphs"]) for i in indexed_map[n]) for n in ("train", "calibration", "test")}
    header = {
        "seed": seed, "max_articles": max_articles, "top_k": top_k,
        "partitions": {n: sorted(ids) for n, ids in split_map.items()},
        "indexed": {n: sorted(ids) for n, ids in indexed_map.items()},
        "withheld": {n: sorted(ids) for n, ids in withheld_map.items()},
        "index_sizes": index_sizes,
    }

    cache_path = Path(cache_path)
    if not rebuild and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("header") == header and "records" in cached:
                print(f"[cache] loaded {len(cached['records'])} records from {cache_path}")
                return cached
            print("[cache] header mismatch, rebuilding")
        except Exception:
            pass

    total_questions = sum(len(para["qas"]) for name in ("train", "calibration", "test") for art_idx in split_map[name] for para in squad_data["data"][art_idx]["paragraphs"])
    print(f"[cache] building: {total_questions} questions across 3 partitions")

    t0 = time.time()
    done = 0
    records = []
    for name in ("train", "calibration", "test"):
        chunks = build_chunks(indexed_map[name], squad_data)
        tfidf = TFIDFIndex(); tfidf.build(chunks)
        bm25 = BM25Index(); bm25.build(chunks)
        retriever = HybridRetriever(tfidf, bm25, top_k)
        indexed_set = set(indexed_map[name])
        for art_idx in split_map[name]:
            article = squad_data["data"][art_idx]
            indexed = art_idx in indexed_set
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    feats = extract_features(qa["question"], retriever)
                    records.append({
                        "qid": qa["id"], "article_idx": art_idx, "partition": name,
                        "indexed": indexed, "is_impossible": bool(qa.get("is_impossible", False)),
                        "features": feats,
                    })
                    done += 1
                    if log_every and done % log_every == 0:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1e-9)
                        print(f"[cache] {done}/{total_questions} eta {(total_questions - done) / max(rate, 1e-9):.0f}s")

    result = {"header": header, "records": records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"[cache] wrote {len(records)} records to {cache_path}")
    return result