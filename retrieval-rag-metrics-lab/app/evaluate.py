from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table

from app.data import load_docs, load_queries
from app.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.search_engines import BM25Search, HybridSearch, TfidfSearch, VectorSearch

console = Console()


def evaluate(k: int = 5) -> pd.DataFrame:
    docs = load_docs()
    queries = load_queries()
    engines = {
        "tfidf": TfidfSearch(docs),
        "bm25": BM25Search(docs),
        "vector": VectorSearch(docs),
        "hybrid": HybridSearch(docs),
        "hybrid_rerank": HybridSearch(docs),
    }
    rows = []
    for name, engine in engines.items():
        recalls, rrs, ndcgs = [], [], []
        for q in queries:
            relevant = set(q["relevant"])
            if name == "hybrid_rerank":
                results = engine.search(q["query"], k=k, rerank=True)
            else:
                results = engine.search(q["query"], k=k)
            recalls.append(recall_at_k(results, relevant, k))
            rrs.append(reciprocal_rank(results, relevant))
            ndcgs.append(ndcg_at_k(results, relevant, k))
        rows.append(
            {
                "engine": name,
                f"Recall@{k}": sum(recalls) / len(recalls),
                "MRR": sum(rrs) / len(rrs),
                f"nDCG@{k}": sum(ndcgs) / len(ndcgs),
            }
        )
    return pd.DataFrame(rows).sort_values(f"nDCG@{k}", ascending=False)


if __name__ == "__main__":
    df = evaluate(k=5)
    table = Table(title="Retrieval Evaluation")
    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)
