import json
from pathlib import Path
from opensearchpy import OpenSearch
from src.config import OPENSEARCH_HOST, INDEX_NAME
from src.data import DOCUMENTS
from src.embeddings import embed_texts, embed_query
from src.hybrid import fuse_results

def get_client():
    return OpenSearch(hosts=[OPENSEARCH_HOST], timeout=30)

def build_mapping(m=16, ef_construction=128, ef_search=100):
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn.algo_param.ef_search": ef_search
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                        "parameters": {
                            "m": m,
                            "ef_construction": ef_construction
                        }
                    }
                }
            }
        }
    }

def create_index(client=None, recreate=True, m=16, ef_construction=128, ef_search=100):
    client = client or get_client()
    if client.indices.exists(index=INDEX_NAME):
        if recreate:
            client.indices.delete(index=INDEX_NAME)
        else:
            return
    client.indices.create(index=INDEX_NAME, body=build_mapping(m, ef_construction, ef_search))

def update_ef_search(ef_search, client=None):
    client = client or get_client()
    client.indices.put_settings(index=INDEX_NAME, body={
        "index": {
            "knn.algo_param.ef_search": ef_search
        }
    })

def load_documents(client=None):
    client = client or get_client()
    texts = [d["title"] + " " + d["text"] for d in DOCUMENTS]
    vectors = embed_texts(texts, normalize=True)
    for doc, vector in zip(DOCUMENTS, vectors):
        body = doc.copy()
        body["embedding"] = vector.tolist()
        client.index(index=INDEX_NAME, id=doc["id"], body=body, refresh=True)

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
    res = client.search(index=INDEX_NAME, body=body)
    return [
        {"id": h["_id"], "score": h["_score"], **{key: val for key, val in h["_source"].items() if key != "embedding"}}
        for h in res["hits"]["hits"]
    ]

def vector_search(query, k=5, client=None, category=None):
    client = client or get_client()
    vector = embed_query(query, normalize=True)[0].tolist()
    knn_query = {
        "knn": {
            "embedding": {
                "vector": vector,
                "k": k
            }
        }
    }
    if category:
        body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_query],
                    "filter": [{"term": {"category": category}}]
                }
            }
        }
    else:
        body = {"size": k, "query": knn_query}

    res = client.search(index=INDEX_NAME, body=body)
    return [
        {"id": h["_id"], "score": h["_score"], **{key: val for key, val in h["_source"].items() if key != "embedding"}}
        for h in res["hits"]["hits"]
    ]

def hybrid_search(query, k=5, alpha=0.5, client=None):
    client = client or get_client()
    bm25 = bm25_search(query, k=k, client=client)
    vector = vector_search(query, k=k, client=client)
    return fuse_results(bm25, vector, alpha=alpha)[:k]
