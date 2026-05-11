import argparse
import pandas as pd
from rich.console import Console
from rich.table import Table
from app.data_loader import load_corpus, load_queries, load_qrels
from app.retriever import TfidfVectorRetriever
from app.metrics import evaluate_query


def run_evaluation(k: int = 5):
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()
    retriever = TfidfVectorRetriever(corpus)

    rows = []
    result_details = {}
    for q in queries:
        results = retriever.search(q["query"], k=k)
        ranked_doc_ids = [r["doc_id"] for r in results]
        metrics = evaluate_query(ranked_doc_ids, qrels.get(q["query_id"], {}), k)
        row = {"query_id": q["query_id"], "query": q["query"], **metrics}
        rows.append(row)
        result_details[q["query_id"]] = results

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if "@" in c]
    summary = df[metric_cols].mean().to_dict()
    return df, summary, result_details


def main():
    parser = argparse.ArgumentParser(description="Evaluate vector/RAG retrieval metrics")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff")
    args = parser.parse_args()

    df, summary, _ = run_evaluation(k=args.k)
    console = Console()
    table = Table(title=f"Vector Search / RAG Retrieval Metrics @ {args.k}")
    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*[f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col]) for col in df.columns])
    console.print(table)
    console.print("\nMean metrics:")
    for name, value in summary.items():
        console.print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
