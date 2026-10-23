from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source: dict
    content_hash: str = ""
    acl: list = field(default_factory=lambda: ["public"])


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
    type: str = "single_source"
    split: str = "dev"
    reviewed: bool = False
