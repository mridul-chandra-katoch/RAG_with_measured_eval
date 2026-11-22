import json
from dataclasses import dataclass, field
from pathlib import Path

def _parse_yaml_simple(text: str) -> dict:
    result = {}
    stack = [result]
    lines = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    
    for line in lines:
        indent = len(line) - len(line.lstrip())
        key_val = line.strip()
        
        while len(stack) > 1 and stack[-1]["_indent"] >= indent:
            stack.pop()
            
        current_dict = stack[-1]
        
        if ": " in key_val:
            key, val = key_val.split(": ", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                current_dict[key] = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            else:
                current_dict[key] = val
        elif key_val.endswith(":"):
            key = key_val[:-1].strip()
            new_dict = {"_indent": indent}
            current_dict[key] = new_dict
            stack.append(new_dict)
            
    def clean(d):
        return {k: clean(v) for k, v in d.items() if k != "_indent"} if isinstance(d, dict) else d
    return clean(result)


@dataclass
class CorpusConfig:
    corpus_name: str
    sources: list
    languages: list = field(default_factory=lambda: ["en"])
    doc_versioning: str = "filename_date"
    
    parsers: dict = field(default_factory=dict)
    chunking_strategy: str = "structure_aware"
    max_tokens: int = 400
    overlap_tokens: int = 50
    
    retrieval_top_k: int = 20
    fusion: str = "rrf"
    reranker_enabled: bool = True
    abstain_mode: str = "learned"
    provider_name: str = "extractive"
    abstain_threshold: float = 0.05
    
    eval_question_set: str = ""
    eval_split_by: str = "document"

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
            
        text = path.read_text(encoding="utf-8")
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_yaml_simple(text)
            
        if "corpus" not in data or "name" not in data["corpus"]:
            raise ValueError("Config must have corpus.name")
        if "sources" not in data["corpus"] or not data["corpus"]["sources"]:
            raise ValueError("Config must have corpus.sources")
            
        corpus = data.get("corpus", {})
        ingest = data.get("ingest", {})
        chunking = data.get("chunking", {})
        retrieval = data.get("retrieval", {})
        generation = data.get("generation", {})
        eval_sec = data.get("eval", {})
        
        abstain_cfg = generation.get("abstain", {})
        threshold = abstain_cfg.get("min_score", 0.05)
        if isinstance(threshold, str) and not threshold.replace('.','',1).isdigit():
            threshold = 0.05  
            
        return cls(
            corpus_name=corpus["name"],
            sources=corpus["sources"],
            languages=corpus.get("languages", ["en"]),
            doc_versioning=corpus.get("doc_versioning", "filename_date"),
            parsers=ingest.get("parsers", {}),
            chunking_strategy=chunking.get("strategy", "structure_aware"),
            max_tokens=int(chunking.get("max_tokens", 400)),
            overlap_tokens=int(chunking.get("overlap_tokens", 50)),
            retrieval_top_k=int(retrieval.get("top_k", 20)),
            fusion=retrieval.get("fusion", "rrf"),
            reranker_enabled=retrieval.get("reranker", {}).get("enabled", True),
            abstain_mode=abstain_cfg.get("mode", "learned"),
            provider_name=generation.get("provider", "extractive"),
            abstain_threshold=float(threshold),
            eval_question_set=eval_sec.get("question_set", ""),
            eval_split_by=eval_sec.get("split_by", "document")
        )

Config = CorpusConfig