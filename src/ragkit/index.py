import math
from collections import Counter


def tokenize(text):
    return [w.strip(".,!?;:\"\'()[]").lower() for w in text.split()
            if w.strip(".,!?;:\"\'()[]")]


class TFIDFIndex:
    def __init__(self):
        self.chunks = []
        self.idf = {}
        self.doc_vectors = []

    def build(self, chunks):
        self.chunks = chunks
        n = len(chunks)
        tokenized = [tokenize(c.text) for c in chunks]
        df = Counter()
        for toks in tokenized:
            df.update(set(toks))
        self.idf = {t: math.log((1 + n) / (1 + df[t])) + 1 for t in df}
        self.doc_vectors = []
        for toks in tokenized:
            tf = Counter(toks)
            vec = {t: tf[t] * self.idf.get(t, 0) for t in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append((vec, norm))

    def search(self, query, top_k=20):
        if not self.chunks:
            return []
        qt = tokenize(query)
        qtf = Counter(qt)
        qvec = {t: qtf[t] * self.idf.get(t, 0) for t in qtf}
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scores = []
        for i, (vec, norm) in enumerate(self.doc_vectors):
            dot = sum(qvec.get(t, 0) * v for t, v in vec.items())
            scores.append((self.chunks[i], dot / (qnorm * norm)))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.docs = []
        self.avgdl = 1.0
        self.idf = {}
        self.doc_freqs = []

    def build(self, chunks):
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, n)
        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self.doc_freqs = [Counter(d) for d in self.docs]

    def search(self, query, top_k=20):
        if not self.chunks:
            return []
        qterms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in qterms:
                if term not in self.doc_freqs[i]:
                    continue
                tf = self.doc_freqs[i][term]
                denom = tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                score += self.idf[term] * (tf * (self.k1 + 1)) / denom
            scores.append((self.chunks[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
