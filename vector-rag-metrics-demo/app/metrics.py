import math
from typing import Dict, List


def recall_at_k(ranked_doc_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """Binary Recall@K: relevant means relevance > 0."""
    relevant = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = set(ranked_doc_ids[:k]) & relevant
    return len(retrieved_relevant) / len(relevant)


def reciprocal_rank(ranked_doc_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document in top K."""
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked_doc_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    """Discounted cumulative gain using graded relevance."""
    total = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = relevant_docs.get(doc_id, 0)
        gain = (2 ** rel) - 1
        discount = math.log2(idx + 1)
        total += gain / discount
    return total


def ndcg_at_k(ranked_doc_ids: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    ideal_rels = sorted(relevant_docs.values(), reverse=True)
    ideal_doc_ids = [f"ideal_{i}" for i in range(len(ideal_rels))]
    ideal_qrels = {doc_id: rel for doc_id, rel in zip(ideal_doc_ids, ideal_rels)}
    ideal_dcg = dcg_at_k(ideal_doc_ids, ideal_qrels, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, relevant_docs, k) / ideal_dcg


def evaluate_query(ranked_doc_ids: List[str], relevant_docs: Dict[str, int], k: int) -> dict:
    return {
        f"Recall@{k}": recall_at_k(ranked_doc_ids, relevant_docs, k),
        f"MRR@{k}": reciprocal_rank(ranked_doc_ids, relevant_docs, k),
        f"nDCG@{k}": ndcg_at_k(ranked_doc_ids, relevant_docs, k),
    }
