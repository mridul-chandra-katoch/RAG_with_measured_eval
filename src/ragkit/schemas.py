from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: dict
    content_hash: str = ""
    acl: list = field(default_factory=lambda: ["public"])
    
    def __post_init__(self):
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if "file" not in self.source:
            raise ValueError("source must contain a 'file' key")

@dataclass
class Citation:
    chunk_id: str
    file: str
    page: Optional[int]
    quote: str

@dataclass
class Answer:
    answer: str
    abstained: bool
    citations: list
    confidence: float
    trace_id: str = ""

@dataclass
class Question:
    q_id: str
    question: str
    gold_chunk_ids: list = field(default_factory=list)
    type: str = "single_source"  # single_source | multi_source | unanswerable | conflicting_version
    split: str = "dev"           # dev | test
    reviewed: bool = False
    author: str = "generated"    # generated | self | other
    
    def __post_init__(self):
        if not self.q_id:
            raise ValueError("q_id cannot be empty")
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if self.type not in ["single_source", "multi_source", "unanswerable", "conflicting_version"]:
            raise ValueError(f"Invalid question type: {self.type}")
        if self.split not in ["dev", "test"]:
            raise ValueError(f"Invalid split: {self.split}")

@dataclass
class RunRecord:
    run_id: str
    corpus: str
    config_hash: str
    data_snapshot_hash: str
    models: dict
    seed: int
    metrics: dict
    per_question: str
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)