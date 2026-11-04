"""Unit tests for ragkit.learned.features on a tiny toy corpus.

Uses the REAL Chunk, TFIDFIndex, BM25Index, HybridRetriever so the tests
exercise the same retrieval code as production.
"""

import unittest

from ragkit.index import BM25Index, TFIDFIndex
from ragkit.retrieve import HybridRetriever
from ragkit.schemas import Chunk

from ragkit.learned.features import (
    FEATURE_COLUMNS,
    extract_features,
    prepare_rankings,
)


def _make_retriever(top_k=3):
    """5-chunk toy corpus where every chunk shares a token with 'apple banana cherry'."""
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="apple banana cherry",
              source={"file": "toy", "page": 0}),
        Chunk(chunk_id="c2", doc_id="d1", text="apple cherry date",
              source={"file": "toy", "page": 1}),
        Chunk(chunk_id="c3", doc_id="d1", text="banana cherry elderberry",
              source={"file": "toy", "page": 2}),
        Chunk(chunk_id="c4", doc_id="d1", text="apple fig grape",
              source={"file": "toy", "page": 3}),
        Chunk(chunk_id="c5", doc_id="d1", text="banana honey ice",
              source={"file": "toy", "page": 4}),
    ]
    tfidf = TFIDFIndex()
    tfidf.build(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    return HybridRetriever(tfidf, bm25, top_k=top_k)


class TestFeatureExtraction(unittest.TestCase):

    def test_feature_columns_order(self):
        self.assertEqual(
            FEATURE_COLUMNS,
            ["f1", "f2", "f3", "f4", "f5a", "f5b", "f6", "f7"],
        )

    def test_fused_matches_retriever_when_both_lists_full_positive(self):
        """When both lists have >= top_k positive-scored results, the
        zero-score-filtered rrf used by the feature code equals the
        unfiltered rrf used by retriever.retrieve()."""
        retriever = _make_retriever(top_k=3)
        query = "apple banana cherry"

        # Sanity: both searches return exactly top_k items, all positive.
        tfidf_raw = retriever.tfidf.search(query, 3)
        bm25_raw = retriever.bm25.search(query, 3)
        self.assertEqual(len(tfidf_raw), 3)
        self.assertEqual(len(bm25_raw), 3)
        self.assertTrue(all(s > 0.0 for _, s in tfidf_raw))
        self.assertTrue(all(s > 0.0 for _, s in bm25_raw))

        _, _, fused_features = prepare_rankings(query, retriever)
        fused_retriever, _ = retriever.retrieve(query)

        self.assertEqual(
            [c.chunk_id for c, _ in fused_features],
            [c.chunk_id for c, _ in fused_retriever],
        )
        for (c1, s1), (c2, s2) in zip(fused_features, fused_retriever):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertAlmostEqual(s1, s2, places=12)

    def test_zero_overlap_query_yields_zero_features(self):
        """A query with no tokens in the corpus yields all 0 features except
        f6 (query length)."""
        retriever = _make_retriever(top_k=3)
        query = "zebra yak xerus"
        feats = extract_features(query, retriever)

        self.assertEqual(len(feats), 8)
        for idx in (0, 1, 2, 3, 4, 5, 7):
            self.assertEqual(
                feats[idx],
                0.0,
                f"{FEATURE_COLUMNS[idx]} should be 0.0 for a zero-overlap query",
            )
        self.assertEqual(feats[6], 3.0)  # query has 3 tokens

    def test_f4_equals_retriever_top_raw(self):
        """f4 must equal the top TF-IDF score retriever.retrieve exposes."""
        retriever = _make_retriever(top_k=3)
        query = "apple banana cherry"
        feats = extract_features(query, retriever)
        _, top_raw = retriever.retrieve(query)
        self.assertAlmostEqual(feats[3], top_raw, places=12)

    def test_prepare_rankings_filters_nonpositive(self):
        """prepare_rankings drops items with score <= 0 from each list."""
        retriever = _make_retriever(top_k=3)
        # 'fig' occurs only in c4; other chunks get zero score.
        query = "fig"
        tfidf_f, bm25_f, fused = prepare_rankings(query, retriever)
        for _, s in tfidf_f:
            self.assertGreater(s, 0.0)
        for _, s in bm25_f:
            self.assertGreater(s, 0.0)
        for _, s in fused:
            self.assertGreater(s, 0.0)


if __name__ == "__main__":
    unittest.main()