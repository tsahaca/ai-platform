from src.data import EVAL_QUERIES
from src.faiss_store import FaissExactStore
from src.opensearch_store import vector_search
from src.metrics import recall_at_k, overlap_count, timer

if __name__ == "__main__":
    k = 5
    faiss_store = FaissExactStore()

    rows = []
    for item in EVAL_QUERIES:
        q = item["query"]
        with timer() as tf:
            faiss_ids = faiss_store.search_ids(q, k=k)
        with timer() as to:
            os_ids = [r["id"] for r in vector_search(q, k=k)]

        rows.append({
            "query": q,
            "faiss_ids": faiss_ids,
            "opensearch_ids": os_ids,
            "recall_vs_faiss": recall_at_k(faiss_ids, os_ids, k),
            "overlap": overlap_count(faiss_ids, os_ids, k),
            "faiss_ms": tf["elapsed_ms"],
            "opensearch_ms": to["elapsed_ms"]
        })

    print("Recall@5 vs FAISS exact baseline")
    for r in rows:
        print("\nQuery:", r["query"])
        print("FAISS:      ", r["faiss_ids"])
        print("OpenSearch: ", r["opensearch_ids"])
        print(f"Recall@5: {r['recall_vs_faiss']:.2f} | overlap={r['overlap']}/5 | faiss={r['faiss_ms']}ms | opensearch={r['opensearch_ms']}ms")
