from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts, normalize=True):
    return get_model().encode(texts, normalize_embeddings=normalize).astype("float32")

def embed_query(query, normalize=True):
    return embed_texts([query], normalize=normalize)
