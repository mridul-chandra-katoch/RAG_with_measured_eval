import re
import uuid
import json
from pathlib import Path

from ragkit.schemas import Answer
from ragkit.learned.features import extract_features
from ragkit.learned import logreg as LR
from ragkit.index import tokenize
from ragkit.providers import ExtractiveProvider, OllamaProvider

STOPWORDS = set("a an the is are was were be been being of in on at to for by with from as what who whom whose which when where why how do does did have has had will would can could should may might must this that these those i you he she it we they me him her us them and or but if then so not no yes there here also any all some each every".split())

def _clean_tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

class RAGPipeline:
    def __init__(self, retriever, abstain_threshold=0.05, abstain_mode="learned", provider_name="extractive", reranker_enabled=True):
        self.retriever = retriever
        self.threshold = abstain_threshold
        self.abstain_mode = abstain_mode
        self.reranker_enabled = reranker_enabled
        self._corpus_tokens = None
        
        self.abstain_data = None
        self.reranker_data = None
        
        if provider_name == "ollama":
            self.provider = OllamaProvider()
        else:
            self.provider = ExtractiveProvider()
        
        if self.abstain_mode == "learned":
            self._load_abstain_model()
            
        if self.reranker_enabled:
            self._load_reranker_model()

    def _load_abstain_model(self):
        model_path = Path("models/abstention_lr.json")
        if model_path.exists():
            self.abstain_data = json.loads(model_path.read_text(encoding="utf-8"))
        else:
            print(f"WARNING: {model_path} not found. Falling back to threshold mode.")
            self.abstain_mode = "threshold"

    def _load_reranker_model(self):
        model_path = Path("models/reranker_xgb.json")
        if model_path.exists():
            self.reranker_data = True
        else:
            print(f"WARNING: {model_path} not found. Disabling reranker.")
            self.reranker_enabled = False

    def _get_corpus_tokens(self):
        if self._corpus_tokens is None:
            tokens = set()
            for c in self.retriever.tfidf.chunks:
                tokens.update(_clean_tokens(c.text))
            self._corpus_tokens = tokens
        return self._corpus_tokens

    def ask(self, question, top_k=5, user_acl=None):
        trace_id = str(uuid.uuid4())
        q_content = set(_clean_tokens(question))
        
        if not q_content:
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        if not (q_content & self._get_corpus_tokens()):
            return Answer("Not found in the documents.", True, [], 0.0, trace_id)
            
        fused, top_raw = self.retriever.retrieve(question, user_acl=user_acl)
        
        # 1. Abstention Check
        use_learned = self.abstain_mode == "learned" and self.abstain_data and len(self.retriever.tfidf.chunks) >= 100
        if use_learned:
            feats = extract_features(question, self.retriever)
            feats_std = LR.standardize([feats], self.abstain_data["means"], self.abstain_data["stds"])[0]
            logit = sum(w * x for w, x in zip(self.abstain_data["weights"], feats_std)) + self.abstain_data["bias"]
            prob = LR.sigmoid(self.abstain_data["platt_a"] * logit + self.abstain_data["platt_b"])
            if prob < 0.5:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
        else:
            if not fused or top_raw < self.threshold:
                return Answer("Not found in the documents.", True, [], 0.0, trace_id)
                
        # 2. Reranker
        if self.reranker_enabled and self.reranker_data and fused:
            import xgboost as xgb
            import numpy as np
            from ragkit.learned.reranker import extract_pair_features
            
            model = xgb.XGBRanker()
            model.load_model("models/reranker_xgb.json")
            
            tfidf_res = self.retriever.tfidf.search(question, self.retriever.top_k)
            bm25_res = self.retriever.bm25.search(question, self.retriever.top_k)
            tfidf_map = {c.chunk_id: s for c, s in tfidf_res}
            bm25_map = {c.chunk_id: s for c, s in bm25_res}
            tfidf_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(tfidf_res)}
            bm25_rank_map = {c.chunk_id: r+1 for r, (c, _) in enumerate(bm25_res)}
            top_bm25 = bm25_res[0][1] if bm25_res else 0.0
            top_tfidf = tfidf_res[0][1] if tfidf_res else 0.0
            
            q_tokens = tokenize(question)
            X_test = []
            for chunk, rrf_score in fused:
                c_tokens = tokenize(chunk.text)
                try: c_pos = int(chunk.chunk_id.split("_p")[1])
                except: c_pos = 0
                raw_f = extract_pair_features(
                    q_tokens, question, c_tokens, chunk.text,
                    bm25_map.get(chunk.chunk_id, 0.0),
                    tfidf_map.get(chunk.chunk_id, 0.0),
                    rrf_score,
                    bm25_rank_map.get(chunk.chunk_id, 0),
                    tfidf_rank_map.get(chunk.chunk_id, 0),
                    c_pos, top_bm25, top_tfidf
                )
                X_test.append(raw_f)
                
            scores = model.predict(np.array(X_test))
            fused = [x for _, x in sorted(zip(scores, fused), key=lambda pair: -pair[0])]
                
        # 3. Generate
        results = fused[:top_k]
        ans = self.provider.generate(question, results)
        ans.trace_id = trace_id
        return ans