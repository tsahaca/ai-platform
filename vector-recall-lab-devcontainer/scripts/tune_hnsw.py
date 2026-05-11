from src.data import EVAL_QUERIES
from src.faiss_store import FaissExactStore
from src.opensearch_store import get_client, create_index, load_documents, vector_search
from src.metrics import recall_at_k, timer
import statistics

PARAMS = [
    {"m": 8, "ef_construction": 64, "ef_search": 20},
    {"m": 16, "ef_construction": 128, "ef_search": 50},
    {"m": 16, "ef_construction": 128, "ef_search": 100},
    {"m": 32, "ef_construction": 256, "ef_search": 100}
]

if __name__ == "__main__":
    k = 5
    client = get_client()
    faiss_store = FaissExactStore()

    print("HNSW tuning comparison")
    for p in PARAMS:
        print("\nTesting:", p)
        create_index(client, recreate=True, **p)
        load_documents(client)

        recalls = []
        latencies = []

        for item in EVAL_QUERIES:
            q = item["query"]
            faiss_ids = faiss_store.search_ids(q, k=k)

            with timer() as t:
                os_ids = [r["id"] for r in vector_search(q, k=k, client=client)]

            recalls.append(recall_at_k(faiss_ids, os_ids, k))
            latencies.append(t["elapsed_ms"])

        print(f"Average Recall@{k}: {statistics.mean(recalls):.2f}")
        print(f"Average OpenSearch Latency: {statistics.mean(latencies):.2f} ms")
