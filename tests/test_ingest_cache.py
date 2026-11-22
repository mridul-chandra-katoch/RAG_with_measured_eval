import unittest
import tempfile
from pathlib import Path
import os
from ragkit.db import MetadataDB
from ragkit.chunk import chunk_documents
import hashlib

class TestIngestCache(unittest.TestCase):
    def test_config_change_triggers_reingest(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            db = MetadataDB(db_path)
            
            folder = Path(d) / "docs"
            folder.mkdir()
            (folder / "test.txt").write_text("Hello world")
            
            # First ingest
            file_hash = hashlib.sha256(b"Hello world").hexdigest()
            doc_id = "test_txt"
            cache_key_1 = hashlib.sha256(f"{file_hash}_1.0_hashA".encode()).hexdigest()
            
            chunks = chunk_documents(folder, 400, 50, "fixed")
            db.add_document_with_chunks(doc_id, "test.txt", cache_key_1, chunks)
            
            self.assertEqual(db.get_cache_key(doc_id), cache_key_1)
            
            # Second ingest with different config
            cache_key_2 = hashlib.sha256(f"{file_hash}_1.0_hashB".encode()).hexdigest()
            self.assertNotEqual(cache_key_1, cache_key_2)

    def test_deleted_file_removed(self):
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            db = MetadataDB(db_path)
            
            folder = Path(d) / "docs"
            folder.mkdir()
            (folder / "test.txt").write_text("Hello")
            
            chunks = chunk_documents(folder, 400, 50, "fixed")
            db.add_document_with_chunks("test_txt", "test.txt", "key1", chunks)
            
            self.assertEqual(len(db.get_all_chunks()), 1)
            db.delete_document("test_txt")
            self.assertEqual(len(db.get_all_chunks()), 0)

    def test_duplicate_chunk_id_regression(self):
        # Regression test for the Phase C chunk-ID collision
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            # Create a markdown file with two identical headings
            (folder / "test.md").write_text("# Section 1\nA B C D\n# Section 2\nA B C D")
            
            chunks = chunk_documents(folder, max_tokens=10, overlap=0, strategy="structure_aware")
            # Even though the text is identical, the chunk IDs must be unique
            chunk_ids = [c.chunk_id for c in chunks]
            self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "Chunk IDs must be unique")

if __name__ == "__main__":
    unittest.main()