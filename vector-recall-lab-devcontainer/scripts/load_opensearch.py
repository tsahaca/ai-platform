import argparse
from src.opensearch_store import get_client, create_index, load_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=100)
    args = parser.parse_args()

    client = get_client()
    print(f"Creating index with m={args.m}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    create_index(client, recreate=True, m=args.m, ef_construction=args.ef_construction, ef_search=args.ef_search)
    print("Loading documents...")
    load_documents(client)
    print("Done.")
