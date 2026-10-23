import hashlib

from ragkit.ingest import load_documents
from ragkit.schemas import Chunk


def chunk_documents(folder, max_tokens=400, overlap=50):
    chunks = []
    for doc in load_documents(folder):
        words = doc["text"].split()
        step = max(1, max_tokens - overlap)
        for start in range(0, max(1, len(words)), step):
            segment = words[start:start + max_tokens]
            if not segment:
                continue
            text = " ".join(segment)
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            doc_id = doc["file"].replace("/", "_").replace("\\", "_").replace(".", "_")
            chunk_id = doc_id + "_p" + str(doc["page"]) + "_c" + str(start // step)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text,
                source={"file": doc["file"], "page": doc["page"]},
                content_hash="sha256:" + h,
            ))
    return chunks
