import json
import urllib.request
import urllib.error
import re
from ragkit.schemas import Answer, Citation

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = set("a an the is are was were be been being of in on at to for by with from as what who whom whose which when where why how do does did have has had will would can could should may might must this that these those i you he she it we they me him her us them and or but if then so not no yes there here also any all some each every".split())

def _clean_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

def _best_sentence(query, chunk_text):
    qtokens = set(_clean_tokens(query))
    sentences = [s for s in SENT_SPLIT.split(chunk_text) if s.strip()]
    if not sentences:
        return chunk_text[:300]
    best_score = -1
    best_sent = sentences[0]
    for s in sentences:
        score = len(qtokens & set(_clean_tokens(s)))
        if score > best_score:
            best_score = score
            best_sent = s
    return best_sent[:300]

class Provider:
    """Base interface for generation providers."""
    def generate(self, query: str, chunks: list) -> Answer:
        raise NotImplementedError

class ExtractiveProvider(Provider):
    """Pulls the best sentence from the top chunk (labeled 'extractive')."""
    def generate(self, query: str, chunks: list) -> Answer:
        if not chunks:
            return Answer("Not found in the documents.", True, [], 0.0)
            
        top_chunk, top_score = chunks[0]
        answer_text = _best_sentence(query, top_chunk.text)
        
        citations = [
            Citation(c.chunk_id, c.source["file"], c.source.get("page"), c.text[:200])
            for c, _ in chunks[:3]
        ]
        
        return Answer(answer_text, False, citations, min(1.0, top_score))

class OllamaProvider(Provider):
    """Calls a local Ollama server via pure urllib (labeled 'pretrained')."""
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def generate(self, query: str, chunks: list) -> Answer:
        if not chunks:
            return Answer("Not found in the documents.", True, [], 0.0)

        context = "\n\n".join([c.text for c, _ in chunks[:3]])
        prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
            f"If the answer is not in the context, say 'Not found in the documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        data = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                answer_text = result.get("response", "").strip()
                if "Not found in the documents" in answer_text:
                    return Answer("Not found in the documents.", True, [], 0.0)
                
                citations = [
                    Citation(c.chunk_id, c.source["file"], c.source.get("page"), c.text[:200])
                    for c, _ in chunks[:3]
                ]
                return Answer(answer_text, False, citations, 0.99)
        except urllib.error.URLError:
            return Answer("Error: Could not connect to Ollama. Is it running?", True, [], 0.0)

class FakeProvider(Provider):
    """Used strictly for unit tests. No network."""
    def __init__(self, response_text="Fake answer."):
        self.response_text = response_text

    def generate(self, query: str, chunks: list) -> Answer:
        return Answer(self.response_text, False, [], 1.0)