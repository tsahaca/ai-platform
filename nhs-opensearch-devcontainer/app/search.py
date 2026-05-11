import sys
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, INDEX_NAME
from .opensearch_client import get_client


def semantic_search(query: str, k: int = 5) -> None:
    client = get_client()
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector = model.encode([query], normalize_embeddings=True).astype("float32")[0].tolist()

    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": k
                }
            }
        },
        "_source": ["title", "section", "url", "chunk_text"]
    }

    response = client.search(index=INDEX_NAME, body=body)
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        print("-" * 80)
        print(f"score: {hit['_score']:.4f}")
        print(f"title: {src['title']}")
        print(f"section: {src['section']}")
        print(f"url: {src['url']}")
        print(src["chunk_text"][:700].replace("\n", " "))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "chest pain when exercising"
    semantic_search(q)
