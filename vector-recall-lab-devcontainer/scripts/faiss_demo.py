import argparse
from src.faiss_store import FaissExactStore
from src.metrics import timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    store = FaissExactStore()
    with timer() as t:
        results = store.search(args.query, args.k)

    print(f"FAISS exact latency: {t['elapsed_ms']} ms")
    for r in results:
        print(f"{r['rank']}. [{r['score']:.4f}] {r['id']} | {r['title']} | {r['category']}")
