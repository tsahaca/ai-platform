from app.search_engines import BM25Search, HybridSearch, TfidfSearch, VectorSearch


def show(title, results):
    print(f"\n=== {title} ===")
    for r in results:
        extra = f" rerank={r.get('rerank_score'):.3f}" if "rerank_score" in r else ""
        print(f"{r['rank']}. {r['id']} | {r['title']} | score={r['score']:.3f}{extra}")


if __name__ == "__main__":
    query = "ADHD medication side effects"
    print(f"Query: {query}")
    show("TF-IDF", TfidfSearch().search(query, 5))
    show("BM25", BM25Search().search(query, 5))
    show("Vector", VectorSearch().search(query, 5))
    show("Hybrid", HybridSearch().search(query, 5))
    show("Hybrid + Rerank", HybridSearch().search(query, 5, rerank=True))
