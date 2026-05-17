from __future__ import annotations

import math


def recall_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    retrieved = {r["id"] for r in results[:k]}
    return len(retrieved & relevant) / max(len(relevant), 1)


def reciprocal_rank(results: list[dict], relevant: set[str]) -> float:
    for i, r in enumerate(results, start=1):
        if r["id"] in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    score = 0.0
    for i, r in enumerate(results[:k], start=1):
        rel = 1.0 if r["id"] in relevant else 0.0
        score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(results: list[dict], relevant: set[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(results, relevant, k) / ideal
