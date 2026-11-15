import unittest
from ragkit.generate import RAGPipeline
from ragkit.index import TFIDFIndex, BM25Index
from ragkit.retrieve import HybridRetriever
from ragkit.schemas import Chunk

class TestGenerateFallback(unittest.TestCase):
    def test_small_corpus_falls_back_to_threshold(self):
        chunks = [
            Chunk("c1", "d1", "Attendance requires 75 percent of all scheduled classes.", {"file": "test.md", "page": 1}),
            Chunk("c2", "d1", "The sky is blue in the document.", {"file": "test.md", "page": 1}),
        ]
        tfidf = TFIDFIndex()
        tfidf.build(chunks)
        bm25 = BM25Index()
        bm25.build(chunks)
        retriever = HybridRetriever(tfidf, bm25, top_k=5)
        
        # Force learned mode, but the index is tiny (2 chunks)
        pipeline = RAGPipeline(retriever, abstain_threshold=0.05, abstain_mode="learned")
        
        ans = pipeline.ask("What is the attendance requirement?")
        
        # The learned model would normally abstain here (out of distribution). 
        # We want the fallback to trigger and answer.
        self.assertFalse(ans.abstained, "Pipeline should fallback to threshold and answer.")
        self.assertNotEqual(ans.answer, "Not found in the documents.")

if __name__ == "__main__":
    unittest.main()