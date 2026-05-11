from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
from sklearn.metrics import ndcg_score


def precision_at_k(relevant_docs: Iterable[str], retrieved_docs: Sequence[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    retrieved_k = retrieved_docs[:k]
    return len(set(retrieved_k) & set(relevant_docs)) / k


def recall_at_k(relevant_docs: Iterable[str], retrieved_docs: Sequence[str], k: int) -> float:
    relevant_set = set(relevant_docs)
    if not relevant_set:
        return 0.0
    retrieved_k = retrieved_docs[:k]
    return len(set(retrieved_k) & relevant_set) / len(relevant_set)


def reciprocal_rank(relevant_docs: Iterable[str], retrieved_docs: Sequence[str]) -> float:
    relevant_set = set(relevant_docs)
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_set:
            return 1 / rank
    return 0.0


def ndcg(relevance_grades: Sequence[int], ranking_scores: Sequence[float]) -> float:
    y_true = np.asarray([relevance_grades])
    y_score = np.asarray([ranking_scores])
    return float(ndcg_score(y_true, y_score))


if __name__ == "__main__":
    relevant = ["doc1", "doc3", "doc5", "doc7"]
    retrieved = ["doc1", "doc2", "doc5", "doc9", "doc3"]

    print("Precision@3:", precision_at_k(relevant, retrieved, 3))
    print("Recall@3:", recall_at_k(relevant, retrieved, 3))
    print("MRR:", reciprocal_rank(relevant, retrieved))
    print("NDCG:", ndcg([3, 2, 0, 1, 0], [0.95, 0.80, 0.70, 0.40, 0.20]))
