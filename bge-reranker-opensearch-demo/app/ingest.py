import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from opensearchpy.helpers import bulk

from app.config import EMBEDDING_MODEL, INDEX_NAME, VECTOR_DIM
from app.opensearch_client import get_client

DATA_FILE = Path("data/sample_documents.json")


def create_index(client):
    if client.indices.exists(INDEX_NAME):
        client.indices.delete(INDEX_NAME)

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=mapping)


def load_docs():
    return json.loads(DATA_FILE.read_text())


def ingest():
    client = get_client()
    create_index(client)

    docs = load_docs()
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in docs]
    vectors = model.encode(texts, normalize_embeddings=True).astype("float32")

    actions = []
    for doc, vector in zip(docs, vectors):
        actions.append({
            "_index": INDEX_NAME,
            "_id": doc["id"],
            "_source": {
                "doc_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "text": doc["text"],
                "embedding": vector.tolist(),
            },
        })

    indexed_count, _ = bulk(client, actions)
    client.indices.refresh(index=INDEX_NAME)

    count_resp = client.count(index=INDEX_NAME)
    indexed_docs = count_resp.get("count", 0)
    print(
        f"Bulk accepted {indexed_count} operations. "
        f"Index now contains {indexed_docs} documents in {INDEX_NAME}."
    )


if __name__ == "__main__":
    ingest()
