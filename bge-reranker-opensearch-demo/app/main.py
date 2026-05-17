import argparse
from rich.console import Console
from rich.table import Table

from app.search import bm25_search, vector_search, hybrid_candidates, rerank

console = Console()


def show(title, rows, score_field="score"):
    table = Table(title=title)
    table.add_column("Rank")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column(score_field)
    table.add_column("Text", overflow="fold")
    for i, r in enumerate(rows, 1):
        table.add_row(str(i), r["title"], r["category"], f"{r[score_field]:.4f}", r["text"][:140])
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="heart attack symptoms")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    console.print(f"[bold]Query:[/bold] {args.query}")
    bm25 = bm25_search(args.query, args.k)
    vector = vector_search(args.query, args.k)
    candidates = hybrid_candidates(args.query, args.k)
    reranked = rerank(args.query, candidates, args.top_k)

    show("BM25 results", bm25)
    show("Vector results", vector)
    show("BGE reranked hybrid results", reranked, "reranker_score")


if __name__ == "__main__":
    main()
