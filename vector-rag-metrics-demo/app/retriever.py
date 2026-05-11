from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfVectorRetriever:
    """
    Lightweight vector-search style retriever.
    For production vector search, replace this with FAISS/OpenSearch embeddings.
    """

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 5):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_matrix).ravel()
        ranked = sorted(zip(self.doc_ids, self.texts, scores), key=lambda x: x[2], reverse=True)
        return [
            {"rank": i + 1, "doc_id": doc_id, "score": float(score), "text": text}
            for i, (doc_id, text, score) in enumerate(ranked[:k])
        ]
