from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

from app.config import EMBEDDING_MODEL, INDEX_NAME, RERANKER_MODEL
from app.opensearch_client import get_client

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    return _reranker


def bm25_search(query: str, k: int = 10):
    client = get_client()
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text"]
            }
        }
    }
    resp = client.search(index=INDEX_NAME, body=body)
    return [format_hit(hit, "bm25") for hit in resp["hits"]["hits"]]


def vector_search(query: str, k: int = 10):
    client = get_client()
    vector = get_embedder().encode([query], normalize_embeddings=True).astype("float32")[0].tolist()
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
    resp = client.search(index=INDEX_NAME, body=body)
    return [format_hit(hit, "vector") for hit in resp["hits"]["hits"]]


def hybrid_candidates(query: str, k_each: int = 10):
    merged = {}
    for row in bm25_search(query, k_each) + vector_search(query, k_each):
        doc_id = row["doc_id"]
        if doc_id not in merged:
            merged[doc_id] = row
            merged[doc_id]["retrievers"] = [row["retriever"]]
        else:
            merged[doc_id]["retrievers"].append(row["retriever"])
            merged[doc_id]["score"] = max(merged[doc_id]["score"], row["score"])
    return list(merged.values())


def rerank(query: str, candidates: list[dict], top_k: int = 5):
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = get_reranker().compute_score(pairs)
    if isinstance(scores, float):
        scores = [scores]
    output = []
    for c, score in zip(candidates, scores):
        item = dict(c)
        item["reranker_score"] = float(score)
        output.append(item)
    return sorted(output, key=lambda x: x["reranker_score"], reverse=True)[:top_k]


def format_hit(hit, retriever):
    src = hit["_source"]
    return {
        "doc_id": src["doc_id"],
        "title": src["title"],
        "category": src["category"],
        "text": src["text"],
        "score": float(hit["_score"]),
        "retriever": retriever,
    }
