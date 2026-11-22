import unittest
from ragkit.retrieve import HybridRetriever
from ragkit.index import TFIDFIndex, BM25Index
from ragkit.schemas import Chunk

class TestRetrieve(unittest.TestCase):
    def setUp(self):
        chunks = [
            Chunk("c1", "d1", "apple banana", {"file": "test"}),
            Chunk("c2", "d1", "banana cherry", {"file": "test"}),
        ]
        tfidf = TFIDFIndex()
        tfidf.build(chunks)
        bm25 = BM25Index()
        bm25.build(chunks)
        self.retriever = HybridRetriever(tfidf, bm25, top_k=2, use_rrf=False)

    def test_bm25_only_order(self):
        fused, _ = self.retriever.retrieve("apple")
        self.assertEqual(fused[0][0].chunk_id, "c1")
        self.assertGreater(fused[0][1], 0.0)

if __name__ == "__main__":
    unittest.main()