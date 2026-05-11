from .config import INDEX_NAME
from .opensearch_client import get_client

DIMENSION = 384

MAPPING = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "condition_id": {"type": "keyword"},
            "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "category": {"type": "keyword"},
            "section": {"type": "keyword"},
            "url": {"type": "keyword"},
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": DIMENSION,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": "cosinesimil"
                }
            }
        }
    }
}


def main(delete_existing: bool = False) -> None:
    client = get_client()
    if client.indices.exists(INDEX_NAME):
        if delete_existing:
            client.indices.delete(INDEX_NAME)
        else:
            print(f"Index already exists: {INDEX_NAME}")
            return
    client.indices.create(index=INDEX_NAME, body=MAPPING)
    print(f"Created index: {INDEX_NAME}")


if __name__ == "__main__":
    main(delete_existing=True)
