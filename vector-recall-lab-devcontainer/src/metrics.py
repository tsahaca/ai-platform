import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    result = {"elapsed_ms": None}
    try:
        yield result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)

def recall_at_k(ground_truth_ids, candidate_ids, k):
    gt = set(ground_truth_ids[:k])
    cand = set(candidate_ids[:k])
    if not gt:
        return 0.0
    return len(gt.intersection(cand)) / len(gt)

def overlap_count(a, b, k):
    return len(set(a[:k]).intersection(set(b[:k])))
