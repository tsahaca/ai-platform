import json
from pathlib import Path
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from src.config import OPENSEARCH_HOST, INDEX_NAME
from src.data import DOCUMENTS
from src.embeddings import embed_texts, embed_query
from src.hybrid import fuse_results


def _connection_help_message() -> str:
    return (
        "Could not connect to OpenSearch at "
        f"{OPENSEARCH_HOST}. "
        "Start OpenSearch and ensure the host is reachable. "
        "You can override the endpoint with OPENSEARCH_HOST, for example: "
        "OPENSEARCH_HOST=http://opensearch:9200 (docker-compose service DNS) or "
        "OPENSEARCH_HOST=http://localhost:9200 (host-mapped port)."
    )


def _run_with_connection_hint(operation):
    try:
        return operation()
    except OpenSearchConnectionError as exc:
        raise RuntimeError(_connection_help_message()) from exc


def get_client():
    return OpenSearch(hosts=[OPENSEARCH_HOST])


def create_index(client=None, recreate=True):
    client = client or get_client()
    index_exists = _run_with_connection_hint(
        lambda: client.indices.exists(index=INDEX_NAME)
    )
    if index_exists:
        if recreate:
            _run_with_connection_hint(lambda: client.indices.delete(index=INDEX_NAME))
        else:
            return

    mapping_path = Path(__file__).resolve().parents[1] / "opensearch" / "index_mapping.json"
    _run_with_connection_hint(
        lambda: client.indices.create(index=INDEX_NAME, body=json.loads(mapping_path.read_text()))
    )


def load_documents(client=None):
    client = client or get_client()
    texts = [d["title"] + " " + d["text"] for d in DOCUMENTS]
    vectors = embed_texts(texts, normalize=True)

    for doc, vector in zip(DOCUMENTS, vectors):
        body = doc.copy()
        body["embedding"] = vector.tolist()
        _run_with_connection_hint(
            lambda: client.index(index=INDEX_NAME, id=doc["id"], body=body, refresh=True)
        )


def bm25_search(query, k=5, client=None):
    client = client or get_client()
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text", "category"]
            }
        }
    }
    res = _run_with_connection_hint(lambda: client.search(index=INDEX_NAME, body=body))
    return [{"id": h["_id"], "score": h["_score"], **{k:v for k,v in h["_source"].items() if k != "embedding"}} for h in res["hits"]["hits"]]


def vector_search(query, k=5, client=None):
    client = client or get_client()
    vector = embed_query(query, normalize=True)[0].tolist()
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": k
                }
            }
        }
    }
    res = _run_with_connection_hint(lambda: client.search(index=INDEX_NAME, body=body))
    return [{"id": h["_id"], "score": h["_score"], **{k:v for k,v in h["_source"].items() if k != "embedding"}} for h in res["hits"]["hits"]]


def hybrid_search(query, k=5, alpha=0.5, client=None):
    client = client or get_client()
    return fuse_results(bm25_search(query, k, client), vector_search(query, k, client), alpha=alpha)[:k]
