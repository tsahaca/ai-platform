from pathlib import Path
import sys

# Ensure `src` is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.opensearch_store import get_client, create_index, load_documents

if __name__ == "__main__":
    try:
        client = get_client()
        print("Creating OpenSearch index...")
        create_index(client, recreate=True)
        print("Loading documents...")
        load_documents(client)
        print("Done.")
    except RuntimeError as exc:
        print(f"OpenSearch setup failed: {exc}")
        raise SystemExit(1)
