import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTIONS = {
    "semantic": "semantic_documents",
    "recommendation": "product_recommendations",
    "rag": "rag_knowledge_base",
    "image": "image_similarity",
    "fraud": "fraud_patterns",
}

app = FastAPI(
    title="Open Source Vector DB + FAISS Demo",
    description="Qdrant stores vectors and metadata. FAISS provides local ANN indexes for fast similarity search demos.",
    version="1.0.0",
)

model: Optional[SentenceTransformer] = None
qdrant: Optional[QdrantClient] = None
faiss_indexes: Dict[str, faiss.Index] = {}
payloads: Dict[str, List[Dict[str, Any]]] = {}
vectors: Dict[str, np.ndarray] = {}


def normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def embed_text(texts: List[str]) -> np.ndarray:
    assert model is not None
    emb = model.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def build_faiss_index(name: str, matrix: np.ndarray) -> None:
    matrix = normalize(matrix)
    index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.add(matrix)
    faiss_indexes[name] = index
    vectors[name] = matrix


def create_collection(name: str, dim: int) -> None:
    assert qdrant is not None
    existing = [c.name for c in qdrant.get_collections().collections]
    if name in existing:
        return
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_collection(collection: str, matrix: np.ndarray, records: List[Dict[str, Any]]) -> None:
    assert qdrant is not None
    create_collection(collection, matrix.shape[1])
    points = [
        PointStruct(id=i, vector=matrix[i].tolist(), payload=records[i])
        for i in range(len(records))
    ]
    qdrant.upsert(collection_name=collection, points=points)


def qdrant_search(collection: str, vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
    assert qdrant is not None
    result = qdrant.search(collection_name=collection, query_vector=vector.tolist(), limit=k)
    return [{"score": r.score, "payload": r.payload} for r in result]


def faiss_search(index_name: str, vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
    index = faiss_indexes[index_name]
    query = normalize(vector.reshape(1, -1))
    scores, ids = index.search(query, k)
    return [
        {"score": float(scores[0][rank]), "payload": payloads[index_name][int(idx)]}
        for rank, idx in enumerate(ids[0])
        if idx >= 0
    ]


def image_feature(path: Path) -> np.ndarray:
    """Lightweight open-source image vector demo: RGB histogram + simple shape/edge hints.

    This avoids requiring a large CLIP model while still demonstrating image similarity
    with vectors. Swap this function with CLIP embeddings for production-grade image search.
    """
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype("float32") / 255.0
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 1), density=True)
        hist_features.extend(hist.tolist())
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    mean_rgb = arr.mean(axis=(0, 1)).tolist()
    feature = np.array(hist_features + [gx, gy] + mean_rgb, dtype="float32")
    return normalize(feature.reshape(1, -1))[0]


def generate_demo_images() -> List[Dict[str, Any]]:
    image_dir = DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("red_circle", "red", "circle"),
        ("red_square", "red", "square"),
        ("blue_circle", "blue", "circle"),
        ("blue_square", "blue", "square"),
        ("green_triangle", "green", "triangle"),
        ("yellow_triangle", "yellow", "triangle"),
    ]
    records = []
    for name, color, shape in specs:
        path = image_dir / f"{name}.png"
        img = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(img)
        if shape == "circle":
            draw.ellipse((50, 50, 206, 206), fill=color)
        elif shape == "square":
            draw.rectangle((50, 50, 206, 206), fill=color)
        else:
            draw.polygon([(128, 40), (220, 210), (36, 210)], fill=color)
        img.save(path)
        records.append({"id": name, "color": color, "shape": shape, "path": str(path)})
    return records


def seed_all() -> None:
    global payloads

    semantic_docs = [
        {"title": "API Gateway", "text": "Gravitee APIM manages APIs, policies, plans, traffic routing, rate limits, and analytics."},
        {"title": "OpenSearch", "text": "OpenSearch supports keyword search, vector search, hybrid search, dashboards, and log analytics."},
        {"title": "Bedrock RAG", "text": "A RAG pipeline retrieves relevant documents and sends context to an LLM for grounded answers."},
        {"title": "Terraform", "text": "Terraform provisions infrastructure using declarative configuration and repeatable plans."},
        {"title": "Solace Messaging", "text": "Solace PubSub+ supports event-driven architecture with topics, queues, and durable subscriptions."},
    ]
    sem_vecs = embed_text([d["text"] for d in semantic_docs])
    payloads["semantic"] = semantic_docs
    build_faiss_index("semantic", sem_vecs)
    upsert_collection(COLLECTIONS["semantic"], sem_vecs, semantic_docs)

    products = [
        {"product": "Trail running shoes", "description": "Lightweight shoes for outdoor running, hiking, and rocky paths."},
        {"product": "Mirrorless camera", "description": "Compact camera with interchangeable lenses, RAW files, and travel photography features."},
        {"product": "Noise-cancelling headphones", "description": "Wireless headphones for flights, office focus, and music listening."},
        {"product": "Backpacking tent", "description": "Ultralight waterproof tent for camping, trekking, and mountain trips."},
        {"product": "Coffee grinder", "description": "Burr grinder for espresso, pour-over, and fresh coffee at home."},
    ]
    prod_vecs = embed_text([p["description"] for p in products])
    payloads["recommendation"] = products
    build_faiss_index("recommendation", prod_vecs)
    upsert_collection(COLLECTIONS["recommendation"], prod_vecs, products)

    rag_docs = [
        {"source": "runbook-1", "text": "For curl timeout exit code 28, check gateway DNS, route availability, backend health, firewall rules, and TLS handshake."},
        {"source": "runbook-2", "text": "For SOAP validation, use a full SOAP envelope XSD when the gateway policy cannot resolve external schema imports."},
        {"source": "runbook-3", "text": "For GitLab CI smoke tests, assert HTTP status codes and response fields using curl and jq."},
        {"source": "runbook-4", "text": "For Terraform Enterprise VCS runs, workspace working directory should match the environment folder containing main.tf."},
    ]
    rag_vecs = embed_text([d["text"] for d in rag_docs])
    payloads["rag"] = rag_docs
    build_faiss_index("rag", rag_vecs)
    upsert_collection(COLLECTIONS["rag"], rag_vecs, rag_docs)

    image_records = generate_demo_images()
    image_vecs = np.vstack([image_feature(Path(r["path"])) for r in image_records]).astype("float32")
    payloads["image"] = image_records
    build_faiss_index("image", image_vecs)
    upsert_collection(COLLECTIONS["image"], image_vecs, image_records)

    fraud_patterns = [
        {"pattern": "normal_small_purchase", "amount": 25, "hour": 14, "country_risk": 0.1, "velocity": 1, "chargeback_history": 0},
        {"pattern": "normal_grocery", "amount": 85, "hour": 18, "country_risk": 0.1, "velocity": 1, "chargeback_history": 0},
        {"pattern": "suspicious_midnight_high_amount", "amount": 2400, "hour": 2, "country_risk": 0.7, "velocity": 6, "chargeback_history": 1},
        {"pattern": "card_testing_velocity", "amount": 3, "hour": 3, "country_risk": 0.6, "velocity": 20, "chargeback_history": 1},
        {"pattern": "foreign_high_value_purchase", "amount": 5200, "hour": 23, "country_risk": 0.9, "velocity": 4, "chargeback_history": 1},
    ]
    fraud_vecs = np.vstack([fraud_vector(p) for p in fraud_patterns]).astype("float32")
    payloads["fraud"] = fraud_patterns
    build_faiss_index("fraud", fraud_vecs)
    upsert_collection(COLLECTIONS["fraud"], fraud_vecs, fraud_patterns)


def fraud_vector(tx: Dict[str, Any]) -> np.ndarray:
    amount = min(float(tx["amount"]), 10000.0) / 10000.0
    hour_angle = 2 * math.pi * float(tx["hour"]) / 24.0
    hour_sin = math.sin(hour_angle)
    hour_cos = math.cos(hour_angle)
    country_risk = float(tx["country_risk"])
    velocity = min(float(tx["velocity"]), 25.0) / 25.0
    chargeback = float(tx["chargeback_history"])
    return normalize(np.array([[amount, hour_sin, hour_cos, country_risk, velocity, chargeback]], dtype="float32"))[0]


@app.on_event("startup")
def startup() -> None:
    global model, qdrant
    model = SentenceTransformer(TEXT_MODEL_NAME)
    qdrant = QdrantClient(url=QDRANT_URL)
    seed_all()


class SearchResponse(BaseModel):
    use_case: str
    engine: str
    query: Dict[str, Any]
    results: List[Dict[str, Any]]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/semantic-search", response_model=SearchResponse)
def semantic_search(q: str = Query("how do I manage api traffic and policies?"), k: int = 3, engine: str = "faiss"):
    v = embed_text([q])[0]
    results = faiss_search("semantic", v, k) if engine == "faiss" else qdrant_search(COLLECTIONS["semantic"], v, k)
    return {"use_case": "Semantic search", "engine": engine, "query": {"q": q}, "results": results}


@app.get("/recommend", response_model=SearchResponse)
def recommend(user_interest: str = Query("I like hiking and mountain travel"), k: int = 3, engine: str = "faiss"):
    v = embed_text([user_interest])[0]
    results = faiss_search("recommendation", v, k) if engine == "faiss" else qdrant_search(COLLECTIONS["recommendation"], v, k)
    return {"use_case": "Recommendation systems", "engine": engine, "query": {"user_interest": user_interest}, "results": results}


@app.get("/rag-chat")
def rag_chat(q: str = Query("How should I debug curl exit code 28 in a gateway smoke test?"), k: int = 2, engine: str = "faiss"):
    v = embed_text([q])[0]
    hits = faiss_search("rag", v, k) if engine == "faiss" else qdrant_search(COLLECTIONS["rag"], v, k)
    context = "\n".join([h["payload"]["text"] for h in hits])
    answer = (
        "This demo uses retrieval plus a simple generated response. "
        "In production, send the retrieved context to an LLM such as Bedrock, Ollama, or vLLM.\n\n"
        f"Question: {q}\n\nRetrieved context:\n{context}\n\nSuggested answer: Start with the highest-scoring runbook context above, then validate DNS, route/backend availability, status codes, and TLS behavior."
    )
    return {"use_case": "Chatbots / RAG pipelines", "engine": engine, "query": {"q": q}, "retrieved_context": hits, "answer": answer}


@app.get("/image-similarity", response_model=SearchResponse)
def image_similarity(image_id: str = Query("red_circle"), k: int = 3, engine: str = "faiss"):
    lookup = {r["id"]: r for r in payloads["image"]}
    if image_id not in lookup:
        return {"use_case": "Image similarity search", "engine": engine, "query": {"error": f"Unknown image_id. Try one of {list(lookup)}"}, "results": []}
    v = image_feature(Path(lookup[image_id]["path"]))
    results = faiss_search("image", v, k) if engine == "faiss" else qdrant_search(COLLECTIONS["image"], v, k)
    return {"use_case": "Image similarity search", "engine": engine, "query": {"image_id": image_id}, "results": results}


@app.get("/fraud-detect")
def fraud_detect(amount: float = 2500, hour: int = 2, country_risk: float = 0.8, velocity: int = 7, chargeback_history: int = 1, k: int = 3, engine: str = "faiss"):
    tx = {"amount": amount, "hour": hour, "country_risk": country_risk, "velocity": velocity, "chargeback_history": chargeback_history}
    v = fraud_vector(tx)
    hits = faiss_search("fraud", v, k) if engine == "faiss" else qdrant_search(COLLECTIONS["fraud"], v, k)
    top_pattern = hits[0]["payload"]["pattern"] if hits else "unknown"
    risk = "high" if any(word in top_pattern for word in ["suspicious", "testing", "foreign_high"]) else "low"
    return {"use_case": "Fraud detection pattern similarity", "engine": engine, "query": tx, "nearest_patterns": hits, "risk_label": risk}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Open Source Vector DB + FAISS demo is running",
        "vector_db": "Qdrant",
        "local_ann_engine": "FAISS HNSW",
        "try": [
            "/semantic-search?q=how do I manage api traffic and policies?",
            "/recommend?user_interest=I like hiking and photography",
            "/rag-chat?q=How do I assert API smoke tests in GitLab CI?",
            "/image-similarity?image_id=red_circle",
            "/fraud-detect?amount=3000&hour=2&country_risk=0.9&velocity=8&chargeback_history=1",
        ],
        "docs": "/docs",
    }
