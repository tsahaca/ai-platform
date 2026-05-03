import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from app.data import load_docs

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class FaissSemanticSearch:
    def __init__(self):
        self.docs = load_docs()
        self.model = SentenceTransformer(MODEL_NAME)
        texts = [d["text"] for d in self.docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product == cosine when normalized
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            {"score": float(scores[0][rank]), **self.docs[int(idx)]}
            for rank, idx in enumerate(ids[0])
        ]

if __name__ == "__main__":
    engine = FaissSemanticSearch()
    for r in engine.search("how can pods connect to solace with TLS hostname", 3):
        print(f"{r['score']:.4f} | {r['id']} | {r['title']} | {r['text']}")
