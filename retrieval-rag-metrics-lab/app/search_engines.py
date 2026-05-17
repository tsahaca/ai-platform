from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.data import load_docs


def tokenize(text: str) -> list[str]:
    return text.lower().replace(":", " ").replace("-", " ").split()


class TfidfSearch:
    def __init__(self, docs=None):
        self.docs = docs or load_docs()
        self.texts = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ q.T).toarray().ravel()
        ranked = np.argsort(scores)[::-1][:k]
        return [{"rank": i + 1, "score": float(scores[idx]), **self.docs[idx]} for i, idx in enumerate(ranked)]


class BM25Search:
    def __init__(self, docs=None):
        self.docs = docs or load_docs()
        tokenized = [tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:k]
        return [{"rank": i + 1, "score": float(scores[idx]), **self.docs[idx]} for i, idx in enumerate(ranked)]


class VectorSearch:
    def __init__(self, docs=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs = docs or load_docs()
        self.model = SentenceTransformer(model_name)
        texts = [d["text"] for d in self.docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            {"rank": rank + 1, "score": float(scores[0][rank]), **self.docs[int(idx)]}
            for rank, idx in enumerate(ids[0])
        ]


class SimpleReranker:
    """Small educational reranker using lexical overlap + phrase bonus.

    In production, replace this with a cross-encoder/BGE/Cohere/Bedrock reranker.
    """

    def score(self, query: str, doc_text: str) -> float:
        q_tokens = tokenize(query)
        d_tokens = tokenize(doc_text)
        q_counter = Counter(q_tokens)
        d_counter = Counter(d_tokens)
        overlap = sum(min(q_counter[t], d_counter[t]) for t in q_counter)
        coverage = overlap / max(len(set(q_tokens)), 1)
        phrase_bonus = 0.5 if query.lower() in doc_text.lower() else 0.0
        rare_token_bonus = sum(1 for t in q_tokens if ":" in t or len(t) > 8) * 0.05
        return coverage + phrase_bonus + rare_token_bonus

    def rerank(self, query: str, candidates: Iterable[dict], k: int = 5):
        rescored = []
        for c in candidates:
            c = dict(c)
            c["rerank_score"] = self.score(query, c["text"])
            rescored.append(c)
        rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return [{**r, "rank": i + 1} for i, r in enumerate(rescored[:k])]


class HybridSearch:
    def __init__(self, docs=None):
        self.docs = docs or load_docs()
        self.bm25 = BM25Search(self.docs)
        self.vector = VectorSearch(self.docs)
        self.reranker = SimpleReranker()

    @staticmethod
    def _normalize(results: list[dict], key="score"):
        values = [r[key] for r in results]
        lo, hi = min(values), max(values)
        if math.isclose(lo, hi):
            return {r["id"]: 1.0 for r in results}
        return {r["id"]: (r[key] - lo) / (hi - lo) for r in results}

    def search(self, query: str, k: int = 5, candidate_k: int = 8, rerank: bool = False):
        bm25_results = self.bm25.search(query, candidate_k)
        vector_results = self.vector.search(query, candidate_k)
        bm25_scores = self._normalize(bm25_results)
        vector_scores = self._normalize(vector_results)

        by_id = {d["id"]: dict(d) for d in self.docs}
        candidate_ids = set(bm25_scores) | set(vector_scores)
        merged = []
        for doc_id in candidate_ids:
            score = 0.5 * bm25_scores.get(doc_id, 0.0) + 0.5 * vector_scores.get(doc_id, 0.0)
            merged.append({"score": score, **by_id[doc_id]})
        merged.sort(key=lambda x: x["score"], reverse=True)
        ranked = [{**r, "rank": i + 1} for i, r in enumerate(merged[:candidate_k])]
        if rerank:
            return self.reranker.rerank(query, ranked, k=k)
        return ranked[:k]
