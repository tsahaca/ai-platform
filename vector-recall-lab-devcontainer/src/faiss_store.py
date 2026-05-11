import faiss
from src.data import DOCUMENTS
from src.embeddings import embed_texts, embed_query

class FaissExactStore:
    def __init__(self):
        self.docs = DOCUMENTS
        texts = [d["title"] + " " + d["text"] for d in self.docs]
        self.embeddings = embed_texts(texts, normalize=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, k: int = 5):
        q = embed_query(query, normalize=True)
        scores, ids = self.index.search(q, k)
        results = []
        for rank, idx in enumerate(ids[0]):
            doc = self.docs[int(idx)].copy()
            doc["score"] = float(scores[0][rank])
            doc["rank"] = rank + 1
            results.append(doc)
        return results

    def search_ids(self, query: str, k: int = 5):
        return [r["id"] for r in self.search(query, k)]
