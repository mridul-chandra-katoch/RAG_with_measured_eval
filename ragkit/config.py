import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    name: str
    source_folder: str
    max_tokens: int = 400
    overlap_tokens: int = 50
    top_k: int = 20
    abstain_threshold: float = 0.05

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
