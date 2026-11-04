import re
import uuid
import json
from pathlib import Path

from ragkit.schemas import Answer, Citation
from ragkit.learned.features import extract_features
from ragkit.learned import logreg as LR

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
    def __init__(self, retriever, abstain_threshold=0.05, abstain_mode="learned"):
        self.retriever = retriever
        self.threshold = abstain_threshold
        self.abstain_mode = abstain_mode
        self._corpus_tokens = None
        self.model_data = None
        
        if self.abstain_mode == "learned":
            self._load_learned_model()

    def _load_learned_model(self):
        model_path = Path("models/abstention_lr.json")
        if model_path.exists():
            self.model_data = json.loads(model_path.read_text(encoding="utf-8"))
        else:
            print(f"WARNING: {model_path} not found. Falling back to threshold mode.")
            self.abstain_mode = "threshold"

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
        
        # Pre-checks (a) and (b)
        if not q_content:
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        if not (q_content & self._get_corpus_tokens()):
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
            
        fused, top_raw = self.retriever.retrieve(question)
        
        if self.abstain_mode == "learned" and self.model_data:
            # Extract features and apply the learned model
            feats = extract_features(question, self.retriever)
            feats_std = LR.standardize([feats], self.model_data["means"], self.model_data["stds"])[0]
            
            # Logistic regression logit: w*x + b
            logit = sum(w * x for w, x in zip(self.model_data["weights"], feats_std)) + self.model_data["bias"]
            
            # Platt scaling
            prob = LR.sigmoid(self.model_data["platt_a"] * logit + self.model_data["platt_b"])
            
            # Abstain if probability of being answerable is < 0.5
            if prob < 0.5:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        else:
            # Legacy threshold rule (c)
            if not fused or top_raw < self.threshold:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
                
        # If we reach here, we are answering
        results = fused[:top_k]
        top_chunk, _ = results[0]
        answer = _best_sentence(question, top_chunk)
        citations = [
            Citation(c.chunk_id, c.source["file"], c.source.get("page"), c.text[:200])
            for c, _ in results[:3]
        ]
        return Answer(answer, False, citations, min(1.0, top_raw), trace_id)