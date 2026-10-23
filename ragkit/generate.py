import re
import uuid

from ragkit.schemas import Answer, Citation

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = set("a an the is are was were be been being of in on at to for by with from as what who whom whose which when where why how do does did have has had will would can could should may might must this that these those i you he she it we they me him her us them and or but if then so not no yes there here also any all some each every".split())


def _clean_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _best_sentence(query, chunk):
    qtokens = set(_clean_tokens(query))
    sentences = [s for s in SENT_SPLIT.split(chunk.text) if s.strip()]
    if not sentences:
        return chunk.text[:300]
    best_score = -1
    best_sent = sentences[0]
    for s in sentences:
        score = len(qtokens & set(_clean_tokens(s)))
        if score > best_score:
            best_score = score
            best_sent = s
    return best_sent[:300]


class RAGPipeline:
    def __init__(self, retriever, abstain_threshold=0.05):
        self.retriever = retriever
        self.threshold = abstain_threshold
        self._corpus_tokens = None

    def _get_corpus_tokens(self):
        if self._corpus_tokens is None:
            tokens = set()
            for c in self.retriever.tfidf.chunks:
                tokens.update(_clean_tokens(c.text))
            self._corpus_tokens = tokens
        return self._corpus_tokens

    def ask(self, question, top_k=5):
        trace_id = str(uuid.uuid4())
        q_content = set(_clean_tokens(question))
        if not q_content:
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        if not (q_content & self._get_corpus_tokens()):
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        fused, top_raw = self.retriever.retrieve(question)
        if not fused or top_raw < self.threshold:
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        results = fused[:top_k]
        top_chunk, _ = results[0]
        answer = _best_sentence(question, top_chunk)
        citations = [
            Citation(c.chunk_id, c.source["file"], c.source.get("page"), c.text[:200])
            for c, _ in results[:3]
        ]
        return Answer(answer, False, citations, min(1.0, top_raw), trace_id)
