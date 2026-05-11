import argparse
from src.opensearch_store import bm25_search, vector_search, hybrid_search
from src.metrics import timer

def print_results(results):
    for r in results:
        score = r.get("hybrid_score", r.get("score", 0.0))
        print(f"{r.get('rank', '-')}. [{score:.4f}] {r['id']} | {r['title']} | {r['category']}")
        print(f"   {r['text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    with timer() as t:
        if args.mode == "bm25":
            results = bm25_search(args.query, args.k)
        elif args.mode == "vector":
            results = vector_search(args.query, args.k)
        else:
            results = hybrid_search(args.query, args.k, args.alpha)

    print(f"Mode: {args.mode}")
    print(f"Latency: {t['elapsed_ms']} ms")
    print_results(results)
