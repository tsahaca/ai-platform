import os, time
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from app.data import load_docs

HOST = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
INDEX = os.getenv("INDEX_NAME", "hybrid_docs")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

client = OpenSearch(HOST, use_ssl=False, verify_certs=False)
model = SentenceTransformer(MODEL_NAME)

def wait_for_opensearch():
    for _ in range(60):
        try:
            if client.ping(): return
        except Exception: pass
        time.sleep(2)
    raise RuntimeError("OpenSearch not ready")

def create_index(dim=384):
    if client.indices.exists(INDEX): client.indices.delete(INDEX)
    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "title": {"type": "text"},
            "text": {"type": "text"},
            "embedding": {"type": "knn_vector", "dimension": dim, "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
        }}
    }
    client.indices.create(INDEX, body=body)

def ingest():
    docs = load_docs()
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    for d, v in zip(docs, vectors):
        client.index(index=INDEX, id=d["id"], body={**d, "embedding": v.tolist()})
    client.indices.refresh(INDEX)

def bm25_search(query, k=3):
    return client.search(index=INDEX, body={"size": k, "query": {"match": {"text": query}}})["hits"]["hits"]

def vector_search(query, k=3):
    q = model.encode([query], normalize_embeddings=True)[0].tolist()
    body = {"size": k, "query": {"knn": {"embedding": {"vector": q, "k": k}}}}
    return client.search(index=INDEX, body=body)["hits"]["hits"]

if __name__ == "__main__":
    wait_for_opensearch(); create_index(); ingest()
    query = "solace tls hostname"
    print("\nBM25 / lexical results")
    for h in bm25_search(query): print(h["_score"], h["_source"]["title"])
    print("\nVector / semantic results")
    for h in vector_search(query): print(h["_score"], h["_source"]["title"])
