import re
import uuid
import json
from pathlib import Path

from ragkit.schemas import Answer
from ragkit.learned.features import extract_features
from ragkit.learned import logreg as LR
from ragkit.providers import ExtractiveProvider, OllamaProvider

STOPWORDS = set("a an the is are was were be been being of in on at to for by with from as what who whom whose which when where why how do does did have has had will would can could should may might must this that these those i you he she it we they me him her us them and or but if then so not no yes there here also any all some each every".split())

def _clean_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

class RAGPipeline:
    def __init__(self, retriever, abstain_threshold=0.05, abstain_mode="learned", provider_name="extractive"):
        self.retriever = retriever
        self.threshold = abstain_threshold
        self.abstain_mode = abstain_mode
        self._corpus_tokens = None
        self.model_data = None
        
        if provider_name == "ollama":
            self.provider = OllamaProvider()
        else:
            self.provider = ExtractiveProvider()
        
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
        
        if not q_content:
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        if not (q_content & self._get_corpus_tokens()):
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
            
        fused, top_raw = self.retriever.retrieve(question)
        
        # Abstention logic
        use_learned = self.abstain_mode == "learned" and self.model_data and len(self.retriever.tfidf.chunks) >= 100
        
        if use_learned:
            feats = extract_features(question, self.retriever)
            feats_std = LR.standardize([feats], self.model_data["means"], self.model_data["stds"])[0]
            logit = sum(w * x for w, x in zip(self.model_data["weights"], feats_std)) + self.model_data["bias"]
            prob = LR.sigmoid(self.model_data["platt_a"] * logit + self.model_data["platt_b"])
            
            if prob < 0.5:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        else:
            if not fused or top_raw < self.threshold:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
                
        # Generate
        results = fused[:top_k]
        ans = self.provider.generate(question, results)
        ans.trace_id = trace_id
        return ans