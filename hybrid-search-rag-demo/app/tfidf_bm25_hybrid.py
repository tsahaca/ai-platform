import math
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from app.data import load_docs

def tokenize(text: str):
    return text.lower().replace(".", "").replace(",", "").split()

class HybridSearch:
    def __init__(self):
        self.docs = load_docs()
        self.tokens = [tokenize(d["text"]) for d in self.docs]
        self.N = len(self.docs)
        self.avgdl = np.mean([len(t) for t in self.tokens])
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.emb = self.model.encode([d["text"] for d in self.docs], normalize_embeddings=True)

    def bm25(self, query: str, k1=1.5, b=0.75):
        q_terms = tokenize(query)
        scores = np.zeros(self.N)
        for term in q_terms:
            df = sum(1 for doc in self.tokens if term in doc)
            if df == 0:
                continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for i, doc in enumerate(self.tokens):
                f = doc.count(term)
                dl = len(doc)
                denom = f + k1 * (1 - b + b * dl / self.avgdl)
                scores[i] += idf * ((f * (k1 + 1)) / denom) if denom else 0
        return scores

    def vector(self, query: str):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return np.dot(self.emb, q)

    @staticmethod
    def minmax(x):
        if np.max(x) == np.min(x):
            return np.zeros_like(x, dtype=float)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def search(self, query: str, k=3, alpha=0.5):
        bm25_scores = self.bm25(query)
        vector_scores = self.vector(query)
        hybrid = alpha * self.minmax(bm25_scores) + (1 - alpha) * self.minmax(vector_scores)
        order = np.argsort(-hybrid)[:k]
        return [{"hybrid": float(hybrid[i]), "bm25": float(bm25_scores[i]), "vector": float(vector_scores[i]), **self.docs[i]} for i in order]

if __name__ == "__main__":
    engine = HybridSearch()
    for r in engine.search("solace tls certificate hostname", k=5, alpha=0.5):
        print(f"hybrid={r['hybrid']:.4f} bm25={r['bm25']:.4f} vector={r['vector']:.4f} | {r['id']} | {r['title']}")
