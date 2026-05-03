import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.faiss_store import FaissVectorStore
from src.metrics import timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    store = FaissVectorStore()
    with timer() as t:
        results = store.search(args.query, args.k)

    print(f"FAISS latency: {t['elapsed_ms']} ms")
    for r in results:
        print(f"{r['rank']}. [{r['score']:.4f}] {r['title']} - {r['text']}")
