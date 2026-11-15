import hashlib
import re
from pathlib import Path

from ragkit.ingest import load_documents
from ragkit.schemas import Chunk

HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)')

def _fixed_chunk(text, file, page, section, max_tokens, overlap):
    """Helper to split text into fixed-size overlapping chunks."""
    chunks = []
    words = text.split()
    step = max(1, max_tokens - overlap)
    
    for start in range(0, max(1, len(words)), step):
        segment = words[start:start + max_tokens]
        if not segment:
            continue
            
        chunk_text = " ".join(segment)
        h = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
        doc_id = file.replace("/", "_").replace("\\", "_").replace(".", "_")
        # Use the hash in the chunk_id to prevent duplicates across sections
        chunk_id = f"{doc_id}_p{page}_{h}"
        
        source = {"file": file, "page": page}
        if section:
            source["section"] = section
            
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=chunk_text,
            source=source,
            content_hash="sha256:" + h,
        ))
    return chunks
def chunk_documents(folder, max_tokens=400, overlap=50, strategy="structure_aware"):
    """Splits documents into retrievable chunks.
    
    Strategies:
      - "fixed": Simple overlapping token windows.
      - "structure_aware": Respects Markdown headings (#) and attaches 
        the heading text to the chunk's metadata.
    """
    chunks = []
    for doc in load_documents(folder):
        text = doc["text"]
        file = doc["file"]
        page = doc["page"]
        
        if strategy == "structure_aware":
            # Split by Markdown-style headings
            sections = re.split(r'\n(?=#{1,6}\s)', text)
            current_section = "General"
            
            for section in sections:
                lines = section.splitlines()
                if lines and HEADING_PATTERN.match(lines[0]):
                    current_section = HEADING_PATTERN.match(lines[0]).group(2).strip()
                    text_to_chunk = "\n".join(lines[1:])
                else:
                    text_to_chunk = section
                    
                chunks.extend(_fixed_chunk(
                    text_to_chunk, file, page, current_section, max_tokens, overlap
                ))
        else:
            # "fixed" strategy
            chunks.extend(_fixed_chunk(
                text, file, page, None, max_tokens, overlap
            ))
            
    return chunks