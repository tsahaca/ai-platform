from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opensearch_store import bm25_search, vector_search, hybrid_search
from src.metrics import timer

TESTS = [
    {"query": "cheap compute", "expected_categories": {"cost", "serverless", "compute"}},
    {"query": "keyword search inverted index", "expected_categories": {"search"}},
    {"query": "containers orchestration", "expected_categories": {"containers"}},
    {"query": "serverless event driven applications", "expected_categories": {"serverless", "architecture"}}
]

def score(results, expected_categories):
    return 0.0 if not results else sum(1 for r in results if r["category"] in expected_categories) / len(results)

if __name__ == "__main__":
    modes = {"bm25": bm25_search, "vector": vector_search, "hybrid": hybrid_search}

    for test in TESTS:
        print("\nQuery:", test["query"])
        for name, fn in modes.items():
            with timer() as t:
                results = fn(test["query"], k=3)
            print(f"  {name:7} relevance={score(results, test['expected_categories']):.2f} latency={t['elapsed_ms']} ms")
