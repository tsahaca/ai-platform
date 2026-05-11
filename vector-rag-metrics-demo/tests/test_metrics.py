from app.metrics import recall_at_k, reciprocal_rank, ndcg_at_k


def test_recall_at_k():
    ranked = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 3, "d3": 2, "d9": 1}
    assert recall_at_k(ranked, qrels, 2) == 1 / 3
    assert recall_at_k(ranked, qrels, 4) == 2 / 3


def test_reciprocal_rank():
    ranked = ["bad1", "good", "bad2"]
    qrels = {"good": 3}
    assert reciprocal_rank(ranked, qrels, 3) == 0.5


def test_ndcg_perfect_is_one():
    ranked = ["d1", "d2", "d3"]
    qrels = {"d1": 3, "d2": 2, "d3": 1}
    assert round(ndcg_at_k(ranked, qrels, 3), 6) == 1.0
