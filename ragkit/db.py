import sqlite3
import json
import datetime
from pathlib import Path
from ragkit.schemas import Chunk

class MetadataDB:
    def __init__(self, db_path="results/ragkit.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    file_hash TEXT,
                    ingested_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    text TEXT,
                    source_json TEXT,
                    content_hash TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)
            conn.commit()

    def get_file_hash(self, doc_id):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_hash FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def delete_document(self, doc_id):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def add_document_with_chunks(self, doc_id, file_path, file_hash, chunks):
        self.delete_document(doc_id)
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (doc_id, file_path, file_hash, ingested_at) VALUES (?, ?, ?, ?)",
                (doc_id, file_path, file_hash, now)
            )
            for c in chunks:
                cursor.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, text, source_json, content_hash) VALUES (?, ?, ?, ?, ?)",
                    (c.chunk_id, c.doc_id, c.text, json.dumps(c.source), c.content_hash)
                )
            conn.commit()

    def get_all_chunks(self):
        chunks = []
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, doc_id, text, source_json, content_hash FROM chunks")
            for row in cursor.fetchall():
                chunks.append(Chunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    text=row[2],
                    source=json.loads(row[3]),
                    content_hash=row[4]
                ))
        return chunks