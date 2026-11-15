import hashlib
import math
from ragkit.index import tokenize

class Embedder:
    """Base interface for all dense embedding models."""
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

class HashingEmbedder(Embedder):
    """A pure-Python dense embedder using the hashing trick.
    
    This is a deterministic, stateless 'dense' implementation. It does not 
    require neural networks or external libraries. It maps tokens to a 
    fixed-size float vector. It is labeled as a pretrained/algorithmic 
    baseline, not a trained model.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            # Hash the token to get an index
            h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            # Hash again for the sign to reduce collisions
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 Normalize the vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec