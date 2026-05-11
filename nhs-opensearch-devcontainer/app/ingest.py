from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import EMBEDDING_MODEL, INDEX_NAME
from .create_index import main as create_index
from .nhs_extract import crawl_condition_documents
from .opensearch_client import get_client


def ingest(limit: int = 20) -> None:
    create_index(delete_existing=False)
    client = get_client()
    model = SentenceTransformer(EMBEDDING_MODEL)
    docs = crawl_condition_documents(limit=limit)

    for doc in tqdm(docs, desc="Indexing NHS condition chunks"):
        doc["embedding"] = model.encode(
            doc["chunk_text"],
            normalize_embeddings=True
        ).astype("float32").tolist()
        client.index(index=INDEX_NAME, id=doc["doc_id"], body=doc, refresh=False)

    client.indices.refresh(index=INDEX_NAME)
    print(f"Indexed {len(docs)} chunk documents into {INDEX_NAME}")


if __name__ == "__main__":
    ingest(limit=20)
