"""Unit tests for ragkit.learned.features."""
import unittest
from ragkit.index import BM25Index, TFIDFIndex
from ragkit.retrieve import HybridRetriever
from ragkit.schemas import Chunk
from ragkit.learned.features import extract_features, FEATURE_COLUMNS

def _make_retriever(top_k=3):
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", text="apple banana cherry",
              source={"file": "toy", "page": 0}),
        Chunk(chunk_id="c2", doc_id="d1", text="apple cherry date",
              source={"file": "toy", "page": 1}),
        Chunk(chunk_id="c3", doc_id="d1", text="banana cherry elderberry",
              source={"file": "toy", "page": 2}),
    ]
    tfidf = TFIDFIndex()
    tfidf.build(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    return HybridRetriever(tfidf, bm25, top_k=top_k)

class TestFeatureExtraction(unittest.TestCase):

    def test_feature_columns_order(self):
        self.assertEqual(len(FEATURE_COLUMNS), 9)
        self.assertEqual(FEATURE_COLUMNS[0], "f1_bm25")
        self.assertEqual(FEATURE_COLUMNS[2], "f3_tfidf")

    def test_zero_overlap_query_yields_zero_features(self):
        retriever = _make_retriever(top_k=3)
        query = "zebra yak xerus"
        feats = extract_features(query, retriever)

        self.assertEqual(len(feats), 9)
        # f1, f2, f3, f4, f5a, f5b, f7 should be 0.0
        for idx in (0, 1, 2, 3, 4, 5, 7):
            self.assertEqual(feats[idx], 0.0, f"{FEATURE_COLUMNS[idx]} should be 0.0")
        # f6 is query length (3 tokens)
        self.assertEqual(feats[6], 3.0)

    def test_scores_at_correct_indices(self):
        retriever = _make_retriever(top_k=3)
        query = "apple banana cherry"
        feats = extract_features(query, retriever)
        
        # f1 is BM25 score (index 0)
        bm25_res = retriever.bm25.search(query, 3)
        self.assertAlmostEqual(feats[0], bm25_res[0][1], places=5)
        
        # f3 is TF-IDF score (index 2)
        tfidf_res = retriever.tfidf.search(query, 3)
        self.assertAlmostEqual(feats[2], tfidf_res[0][1], places=5)

if __name__ == "__main__":
    unittest.main()