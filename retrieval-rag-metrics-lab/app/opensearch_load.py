from __future__ import annotations

import os
import time

from opensearchpy import OpenSearch

from app.data import load_docs

INDEX = "retrieval_lab_docs"


def client():
    url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    user = os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PASSWORD")
    auth = (user, password) if user and password else None
    return OpenSearch(url, http_auth=auth, verify_certs=False, ssl_show_warn=False)


def wait_for_opensearch(os_client, timeout=60):
    start = time.time()
    last_error = "no response yet"
    while time.time() - start < timeout:
        try:
            if os_client.ping():
                return
            last_error = "ping returned False"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"OpenSearch did not become ready within {timeout}s. Last check: {last_error}")


def main():
    os_client = client()
    wait_for_opensearch(os_client)
    if os_client.indices.exists(INDEX):
        os_client.indices.delete(INDEX)
    os_client.indices.create(
        INDEX,
        body={
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "category": {"type": "keyword"},
                    "text": {"type": "text"},
                }
            },
        },
    )
    for doc in load_docs():
        os_client.index(index=INDEX, id=doc["id"], body=doc, refresh=True)
    print(f"Loaded {len(load_docs())} docs into {INDEX}")


if __name__ == "__main__":
    main()
