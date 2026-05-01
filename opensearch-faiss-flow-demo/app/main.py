import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from opensearchpy import OpenSearch
from PIL import Image, ImageDraw
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("INDEX_NAME", "multimodal-faiss-demo")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "clip-ViT-B-32")
VECTOR_DIMENSION = 512

app = FastAPI(
    title="OpenSearch + FAISS multimodal vector-search demo",
    description="Text/image/document -> embedding -> OpenSearch knn_vector -> FAISS ANN -> nearest results",
)

client = OpenSearch(OPENSEARCH_URL, use_ssl=False, verify_certs=False)
model: SentenceTransformer | None = None


class TextIngestRequest(BaseModel):
    title: str
    text: str
    metadata: dict[str, Any] = {}


class DocumentIngestRequest(BaseModel):
    title: str
    document_text: str
    metadata: dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_type: Literal["all", "text", "image", "document"] = "all"


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model


def normalize(vector: np.ndarray) -> list[float]:
    vector = vector.astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def embed_text(text: str) -> list[float]:
    embedding = get_model().encode(text, convert_to_numpy=True)
    return normalize(embedding)


def embed_image(image_bytes: bytes) -> list[float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = get_model().encode(image, convert_to_numpy=True)
    return normalize(embedding)


def wait_for_opensearch() -> None:
    for _ in range(60):
        try:
            if client.ping():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("OpenSearch is not reachable")


def create_index() -> None:
    if client.indices.exists(index=INDEX_NAME):
        return

    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "doc_type": {"type": "keyword"},
                "source_path": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 128,
                        },
                    },
                },
            }
        },
    }
    client.indices.create(index=INDEX_NAME, body=mapping)


def index_document(
    *,
    doc_type: str,
    title: str,
    content: str,
    embedding: list[float],
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    doc_id = str(uuid.uuid4())
    client.index(
        index=INDEX_NAME,
        id=doc_id,
        refresh=True,
        body={
            "title": title,
            "content": content,
            "doc_type": doc_type,
            "source_path": source_path,
            "metadata": metadata or {},
            "embedding": embedding,
        },
    )
    return doc_id


@app.on_event("startup")
def startup() -> None:
    wait_for_opensearch()
    create_index()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "flow": [
            "Text/image/document",
            "Embedding model converts it to vector",
            "OpenSearch stores vector in knn_vector field",
            "FAISS HNSW builds ANN index over vectors",
            "User query is embedded",
            "OpenSearch + FAISS finds nearest vectors",
            "Returns matching text/image/document",
        ],
        "docs": "http://localhost:8000/docs",
        "index": INDEX_NAME,
        "model": MODEL_NAME,
    }


@app.post("/admin/reset")
def reset_index() -> dict[str, str]:
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    create_index()
    return {"status": "reset", "index": INDEX_NAME}


@app.post("/ingest/text")
def ingest_text(request: TextIngestRequest) -> dict[str, Any]:
    vector = embed_text(f"{request.title}\n{request.text}")
    doc_id = index_document(
        doc_type="text",
        title=request.title,
        content=request.text,
        embedding=vector,
        metadata=request.metadata,
    )
    return {"id": doc_id, "doc_type": "text", "dimension": len(vector)}


@app.post("/ingest/document")
def ingest_document(request: DocumentIngestRequest) -> dict[str, Any]:
    vector = embed_text(f"{request.title}\n{request.document_text}")
    doc_id = index_document(
        doc_type="document",
        title=request.title,
        content=request.document_text,
        embedding=vector,
        metadata=request.metadata,
    )
    return {"id": doc_id, "doc_type": "document", "dimension": len(vector)}


@app.post("/ingest/image")
async def ingest_image(
    title: str = Form(...),
    caption: str = Form(""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    image_bytes = await file.read()
    vector = embed_image(image_bytes)
    output_dir = Path("/app/samples/uploaded")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}-{file.filename}"
    image_path = output_dir / safe_name
    image_path.write_bytes(image_bytes)

    doc_id = index_document(
        doc_type="image",
        title=title,
        content=caption or title,
        embedding=vector,
        source_path=str(image_path),
        metadata={"filename": file.filename, "content_type": file.content_type},
    )
    return {"id": doc_id, "doc_type": "image", "dimension": len(vector), "source_path": str(image_path)}


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    query_vector = embed_text(request.query)

    knn_query: dict[str, Any] = {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": request.k,
            }
        }
    }

    if request.doc_type != "all":
        query = {
            "bool": {
                "must": [knn_query],
                "filter": [{"term": {"doc_type": request.doc_type}}],
            }
        }
    else:
        query = knn_query

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": request.k,
            "query": query,
            "_source": ["title", "content", "doc_type", "source_path", "metadata"],
        },
    )

    hits = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        hits.append(
            {
                "score": hit["_score"],
                "id": hit["_id"],
                "doc_type": source.get("doc_type"),
                "title": source.get("title"),
                "content": source.get("content"),
                "source_path": source.get("source_path"),
                "metadata": source.get("metadata", {}),
            }
        )

    return {"query": request.query, "k": request.k, "results": hits}


def make_demo_image(path: Path, label: str, background: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (512, 320), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 472, 280), outline=(255, 255, 255), width=6)
    draw.text((70, 140), label, fill=(255, 255, 255))
    image.save(path)


@app.post("/seed")
def seed() -> dict[str, Any]:
    samples_dir = Path("/app/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    text_docs = [
        ("Cloud migration", "AWS migration project using Terraform, EKS, OpenSearch, and observability."),
        ("Italian dinner", "Homemade pasta with tomato sauce, basil, garlic, and olive oil."),
        ("Mountain travel", "A scenic mountain lake trip with hiking, cold weather, and alpine views."),
    ]
    document_docs = [
        ("RAG architecture note", "A retrieval augmented generation pipeline embeds documents, stores vectors, retrieves context, and sends top chunks to an LLM."),
        ("Fraud pattern note", "Suspicious payment behavior includes unusual transaction amount, new device, high velocity, and distance from normal location."),
    ]
    image_specs = [
        ("red sports car", "A red sports car on a road", (180, 40, 40)),
        ("blue mountain lake", "A blue alpine lake near mountains", (40, 80, 180)),
        ("green forest trail", "A green hiking trail in a forest", (40, 140, 70)),
    ]

    created = []
    for title, text in text_docs:
        created.append(index_document(doc_type="text", title=title, content=text, embedding=embed_text(f"{title}\n{text}")))

    for title, text in document_docs:
        created.append(index_document(doc_type="document", title=title, content=text, embedding=embed_text(f"{title}\n{text}")))

    for title, caption, color in image_specs:
        path = samples_dir / f"{title.replace(' ', '-')}.png"
        make_demo_image(path, title, color)
        created.append(
            index_document(
                doc_type="image",
                title=title,
                content=caption,
                embedding=embed_image(path.read_bytes()),
                source_path=str(path),
                metadata={"caption": caption},
            )
        )

    return {"created": len(created), "ids": created}


@app.get("/debug/index-mapping")
def get_mapping() -> dict[str, Any]:
    if not client.indices.exists(index=INDEX_NAME):
        raise HTTPException(status_code=404, detail="index does not exist")
    return client.indices.get_mapping(index=INDEX_NAME)


@app.get("/debug/faiss-field")
def faiss_field() -> dict[str, Any]:
    return {
        "field": "embedding",
        "type": "knn_vector",
        "dimension": VECTOR_DIMENSION,
        "method": {
            "name": "hnsw",
            "engine": "faiss",
            "space_type": "innerproduct",
            "parameters": {"m": 16, "ef_construction": 128},
        },
        "note": "Vectors are normalized, so inner product behaves like cosine similarity.",
    }
