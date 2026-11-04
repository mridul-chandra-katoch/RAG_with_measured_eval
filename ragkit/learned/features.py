"""Feature extraction, article splitting, and caching for the learned
abstention classifier.

Design constraints honored here:
- Imports only ragkit.index, ragkit.retrieve, ragkit.schemas. Never
  ragkit.chunk or ragkit.ingest (they import pdfminer at module level).
- extract_features takes only (query_text, retriever); it never sees the
  label, gold answer, is_impossible flag, or article indexed/withheld status.
- Zero-score filtering is applied before computing any feature, so a
  zero-overlap query does not look like perfect retriever agreement.
"""

import json
import random
import time
from pathlib import Path

from ragkit.index import BM25Index, TFIDFIndex, tokenize
from ragkit.retrieve import HybridRetriever, rrf
from ragkit.schemas import Chunk


# 8 input columns, in this exact order.
FEATURE_COLUMNS = ["f1", "f2", "f3", "f4", "f5a", "f5b", "f6", "f7"]

# Ablation groups. f5 is one group with two columns (f5a, f5b).
FEATURE_GROUPS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
FEATURE_GROUP_COLUMNS = {
    "f1": [0],
    "f2": [1],
    "f3": [2],
    "f4": [3],
    "f5": [4, 5],
    "f6": [6],
    "f7": [7],
}


# ---------------------------------------------------------------------------
# Splits and index assignment
# ---------------------------------------------------------------------------

def split_articles(num_articles, max_articles, seed):
    """Split SQuAD article indices into train/calibration/test by article.

    Inputs:
        num_articles: total articles in the SQuAD file (442 for train-v2.0).
        max_articles: number of articles to sample (N).
        seed: RNG seed; the same seed yields the same split.
    Outputs:
        dict with keys 'train', 'calibration', 'test', each a sorted list of
        original article indices. The three sets are pairwise disjoint and
        their union has size max_articles.
    Assumptions:
        Ratios are 60/15/25 of N (rounded; the remainder goes to test).
        Each partition must end up with at least 2 articles (needed later for
        indexed/withheld). Raises ValueError otherwise.
    """
    if max_articles < 10:
        raise ValueError(f"max_articles must be >= 10, got {max_articles}")
    if max_articles > num_articles:
        raise ValueError(
            f"max_articles ({max_articles}) > num_articles ({num_articles})"
        )
    rng = random.Random(seed)
    sampled = rng.sample(range(num_articles), max_articles)
    n_train = round(0.60 * max_articles)
    n_cal = round(0.15 * max_articles)
    n_test = max_articles - n_train - n_cal
    if n_train < 2 or n_cal < 2 or n_test < 2:
        raise ValueError(
            f"Split too small: train={n_train}, calibration={n_cal}, "
            f"test={n_test}; each must be >= 2."
        )
    train = sorted(sampled[:n_train])
    cal = sorted(sampled[n_train:n_train + n_cal])
    test = sorted(sampled[n_train + n_cal:])
    # Hard assertions: disjointness + size.
    if set(train) & set(cal) or set(train) & set(test) or set(cal) & set(test):
        raise AssertionError("Article splits are not pairwise disjoint")
    if len(train) != n_train or len(cal) != n_cal or len(test) != n_test:
        raise AssertionError("Split sizes do not match requested ratios")
    return {"train": train, "calibration": cal, "test": test}


def assign_indexed_withheld(article_ids, seed, partition_name, indexed_fraction=0.7):
    """Split one partition's articles into 'indexed' and 'withheld' subsets.

    Inputs:
        article_ids: sorted list of original article indices in one partition.
        seed: base RNG seed.
        partition_name: 'train', 'calibration', or 'test'.
        indexed_fraction: approximate fraction to mark indexed (~0.7).
    Outputs:
        (indexed, withheld) as sorted lists. Both are non-empty when
        len(article_ids) >= 2.
    Assumptions:
        Uses random.Random(f"{seed}:{partition_name}") for determinism.
        Raises ValueError if len(article_ids) < 2.
    """
    if len(article_ids) < 2:
        raise ValueError(
            f"Partition '{partition_name}' needs >= 2 articles, "
            f"got {len(article_ids)}"
        )
    rng = random.Random(f"{seed}:{partition_name}")
    shuffled = list(article_ids)
    rng.shuffle(shuffled)
    n_indexed = max(1, min(len(shuffled) - 1, round(indexed_fraction * len(shuffled))))
    indexed = sorted(shuffled[:n_indexed])
    withheld = sorted(shuffled[n_indexed:])
    return indexed, withheld


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------

def build_chunks(article_indices, squad_data):
    """Build one Chunk per SQuAD paragraph from the given articles.

    Inputs:
        article_indices: iterable of original article indices into
            squad_data['data'].
        squad_data: parsed SQuAD 2.0 JSON.
    Outputs:
        list[Chunk]. chunk_id = f"a{article_idx}_p{para_idx}",
        doc_id = f"a{article_idx}", source = {"file": title, "page": para_idx}.
    Assumptions:
        article_idx is the position in the ORIGINAL SQuAD file, not after
        sampling. chunk_ids are unique within the returned list.
    """
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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def prepare_rankings(query_text, retriever):
    """Apply zero-score filtering and return the fused rankings.

    Inputs:
        query_text: str.
        retriever: HybridRetriever with .tfidf, .bm25, .top_k.
    Outputs:
        (tfidf_results, bm25_results, fused) where each list contains only
        strictly-positive-scored (Chunk, score) pairs. Lists may be empty.
    Assumptions:
        Positive scores form a prefix of each search result, so filtering
        by score > 0 preserves surviving ranks (<repo_facts> item 3).
    """
    top_k = retriever.top_k
    tfidf_raw = retriever.tfidf.search(query_text, top_k)
    bm25_raw = retriever.bm25.search(query_text, top_k)
    tfidf_results = [(c, s) for c, s in tfidf_raw if s > 0.0]
    bm25_results = [(c, s) for c, s in bm25_raw if s > 0.0]
    fused = rrf(tfidf_results, bm25_results)
    return tfidf_results, bm25_results, fused


def extract_features(query_text, retriever):
    """Compute the 8 abstention features for one query against one retriever.

    Inputs:
        query_text: str.
        retriever: HybridRetriever.
    Outputs:
        list of 8 floats in FEATURE_COLUMNS order:
          f1  top fused RRF score
          f2  rank-1 minus rank-2 fused score (missing rank -> 0.0)
          f3  top BM25 raw score
          f4  top TF-IDF raw score
          f5a 1 if BM25 rank-1 == TF-IDF rank-1, else 0
          f5b Jaccard of the two top-5 chunk-id sets
          f6  query length in ragkit.index.tokenize tokens
          f7  fraction of unique query tokens present in the fused top-1 chunk
    Assumptions:
        Takes no label information. Zero-score filtering applied first.
    """
    tfidf_results, bm25_results, fused = prepare_rankings(query_text, retriever)

    f1 = fused[0][1] if fused else 0.0

    if len(fused) >= 2:
        f2 = fused[0][1] - fused[1][1]
    elif len(fused) == 1:
        f2 = fused[0][1] - 0.0
    else:
        f2 = 0.0

    f3 = bm25_results[0][1] if bm25_results else 0.0
    f4 = tfidf_results[0][1] if tfidf_results else 0.0

    if bm25_results and tfidf_results:
        f5a = 1.0 if bm25_results[0][0].chunk_id == tfidf_results[0][0].chunk_id else 0.0
    else:
        f5a = 0.0

    bm25_top5 = {c.chunk_id for c, _ in bm25_results[:5]}
    tfidf_top5 = {c.chunk_id for c, _ in tfidf_results[:5]}
    union = bm25_top5 | tfidf_top5
    f5b = (len(bm25_top5 & tfidf_top5) / len(union)) if union else 0.0

    qtokens = tokenize(query_text)
    f6 = float(len(qtokens))

    if fused and qtokens:
        unique_q = set(qtokens)
        top1_tokens = set(tokenize(fused[0][0].text))
        f7 = len(unique_q & top1_tokens) / len(unique_q)
    else:
        f7 = 0.0

    return [f1, f2, f3, f4, f5a, f5b, f6, f7]


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def _count_index_chunks(article_indices, squad_data):
    """Number of paragraph chunks produced by build_chunks for these articles."""
    return sum(len(squad_data["data"][i]["paragraphs"]) for i in article_indices)


def build_or_load_cache(
    squad_data,
    split_map,
    indexed_map,
    withheld_map,
    top_k,
    seed,
    max_articles,
    cache_path,
    rebuild=False,
    log_every=500,
):
    """Build (or load) the per-question feature cache for all partitions.

    Inputs:
        squad_data: parsed SQuAD JSON.
        split_map: {partition: sorted article indices}.
        indexed_map: {partition: sorted indexed article indices}.
        withheld_map: {partition: sorted withheld article indices}.
        top_k: retrieval top_k used to build each retriever.
        seed, max_articles: cache header fields.
        cache_path: Path to the JSON cache file.
        rebuild: if True, ignore any existing cache and rebuild.
        log_every: print a progress line every N questions.
    Outputs:
        dict {'header': ..., 'records': [...]}. Each record is
        {qid, article_idx, partition, indexed, is_impossible, features:[8 floats]}.
    Assumptions:
        Runs retrieval exactly once per question. Header includes partition
        article-id lists and per-partition index sizes; on any mismatch the
        cache is rebuilt.
    """
    index_sizes = {
        name: _count_index_chunks(indexed_map[name], squad_data)
        for name in ("train", "calibration", "test")
    }
    header = {
        "seed": seed,
        "max_articles": max_articles,
        "top_k": top_k,
        "partitions": {name: sorted(ids) for name, ids in split_map.items()},
        "indexed": {name: sorted(ids) for name, ids in indexed_map.items()},
        "withheld": {name: sorted(ids) for name, ids in withheld_map.items()},
        "index_sizes": index_sizes,
    }

    cache_path = Path(cache_path)
    if not rebuild and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("header") == header and "records" in cached:
                print(
                    f"[cache] loaded {len(cached['records'])} records from {cache_path}"
                )
                return cached
            print("[cache] header mismatch, rebuilding")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[cache] could not read {cache_path} ({exc}), rebuilding")

    total_questions = sum(
        len(para["qas"])
        for name in ("train", "calibration", "test")
        for art_idx in split_map[name]
        for para in squad_data["data"][art_idx]["paragraphs"]
    )
    print(f"[cache] building: {total_questions} questions across 3 partitions")

    t0 = time.time()
    done = 0
    records = []
    for name in ("train", "calibration", "test"):
        chunks = build_chunks(indexed_map[name], squad_data)
        tfidf = TFIDFIndex()
        tfidf.build(chunks)
        bm25 = BM25Index()
        bm25.build(chunks)
        retriever = HybridRetriever(tfidf, bm25, top_k)
        indexed_set = set(indexed_map[name])
        for art_idx in split_map[name]:
            article = squad_data["data"][art_idx]
            indexed = art_idx in indexed_set
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    feats = extract_features(qa["question"], retriever)
                    records.append({
                        "qid": qa["id"],
                        "article_idx": art_idx,
                        "partition": name,
                        "indexed": indexed,
                        "is_impossible": bool(qa.get("is_impossible", False)),
                        "features": feats,
                    })
                    done += 1
                    if log_every and done % log_every == 0:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1e-9)
                        remaining = (total_questions - done) / max(rate, 1e-9)
                        print(
                            f"[cache] {done}/{total_questions} "
                            f"({100.0 * done / total_questions:.1f}%) "
                            f"eta {remaining:.0f}s"
                        )

    result = {"header": header, "records": records}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"[cache] wrote {len(records)} records to {cache_path}")
    return result