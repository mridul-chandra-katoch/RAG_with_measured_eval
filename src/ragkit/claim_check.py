import re

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _normalize(text: str) -> str:
    """Lowercases and collapses all whitespace to single spaces."""
    return " ".join(text.lower().split())

def check_claims(answer_text: str, citations: list, chunks: list) -> dict:
    """
    Checks if the answer text is supported by the cited chunks.
    
    Returns:
        dict with keys:
        - "sentences": list of dicts (text, label, matched_chunk)
        - "overall_label": "cited" or "unverified"
    """
    # Map chunk_id to normalized text for fast lookup
    chunk_map = {c.chunk_id: _normalize(c.text) for c in chunks}
    
    sentences = [s.strip() for s in SENT_SPLIT.split(answer_text) if s.strip()]
    if not sentences:
        return {"sentences": [], "overall_label": "unverified"}

    results = []
    overall_verified = True

    for sent in sentences:
        norm_sent = _normalize(sent)
        words = norm_sent.split()
        
        # We need at least 6 words to form a valid quote
        if len(words) < 6:
            results.append({"text": sent, "label": "unverified", "matched_chunk": None})
            overall_verified = False
            continue

        found_match = False
        
        # Generate all 6-word windows from the sentence
        windows = [" ".join(words[i:i+6]) for i in range(len(words) - 5)]
        
        for cite in citations:
            chunk_text = chunk_map.get(cite.chunk_id, "")
            if not chunk_text:
                continue
                
            for window in windows:
                if window in chunk_text:
                    results.append({
                        "text": sent,
                        "label": "cited",
                        "matched_chunk": cite.chunk_id,
                        "quote": window
                    })
                    found_match = True
                    break
            if found_match:
                break

        if not found_match:
            results.append({"text": sent, "label": "unverified", "matched_chunk": None})
            overall_verified = False

    return {
        "sentences": results,
        "overall_label": "cited" if overall_verified else "unverified"
    }