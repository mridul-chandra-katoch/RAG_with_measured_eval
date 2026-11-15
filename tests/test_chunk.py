import unittest
from pathlib import Path
import tempfile
from ragkit.chunk import chunk_documents

class TestChunking(unittest.TestCase):
    def test_chunking_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.txt"
            p.write_text("word " * 500, encoding="utf-8")
            
            chunks = chunk_documents(d, max_tokens=100, overlap=10)
            self.assertTrue(len(chunks) > 1)
            self.assertIn("word", chunks[0].text)
            self.assertEqual(len(chunks[0].text.split()), 100)

if __name__ == "__main__":
    unittest.main()