import unittest
from pathlib import Path
import tempfile
from ragkit.ingest import load_documents

class TestIngest(unittest.TestCase):
    def test_load_txt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.txt"
            p.write_text("Hello world. This is a test document.", encoding="utf-8")
            
            docs = list(load_documents(d))
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["file"], "test.txt")
            self.assertIn("Hello world", docs[0]["text"])
            self.assertEqual(docs[0]["page"], 1)

if __name__ == "__main__":
    unittest.main()