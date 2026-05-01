# OpenSearch + FAISS Vector Flow Demo

This project demonstrates the full flow:

```text
Text, image, document
        ↓
Embedding model converts it to vector
        ↓
OpenSearch stores vector in knn_vector field
        ↓
FAISS builds ANN index over vectors
        ↓
User query is embedded
        ↓
OpenSearch + FAISS finds nearest vectors
        ↓
Returns matching text/image/document
```

## Stack

- **OpenSearch 2.13** as the search database
- **OpenSearch k-NN plugin** with `engine: faiss`
- **FAISS HNSW** as the approximate-nearest-neighbor index engine
- **FastAPI** demo service
- **SentenceTransformers CLIP model**: `clip-ViT-B-32`
  - Text/document content is embedded with CLIP text encoder
  - Images are embedded with CLIP image encoder

> First startup downloads the embedding model, so it can take a little longer.

## Run

```bash
docker compose up --build
```

Open the API docs:

```text
http://localhost:8000/docs
```

Check OpenSearch:

```bash
curl http://localhost:9200
```

## Seed demo data

```bash
curl -X POST http://localhost:8000/seed | jq
```

This creates sample text, document, and image records.

## Search across text, images, and documents

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "mountain lake hiking trip",
    "k": 5,
    "doc_type": "all"
  }' | jq
```

Search only images:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "blue alpine lake",
    "k": 3,
    "doc_type": "image"
  }' | jq
```

Search only documents:

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "retrieval augmented generation architecture",
    "k": 3,
    "doc_type": "document"
  }' | jq
```

## Ingest text

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "OpenSearch vector search",
    "text": "OpenSearch stores embeddings in knn_vector fields and uses FAISS for ANN search.",
    "metadata": {"source": "manual"}
  }' | jq
```

## Ingest document text

```bash
curl -X POST http://localhost:8000/ingest/document \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Architecture document",
    "document_text": "The application generates embeddings, stores them in OpenSearch, and uses FAISS HNSW to return nearest neighbors.",
    "metadata": {"file": "architecture.txt"}
  }' | jq
```

## Ingest an image

```bash
curl -X POST http://localhost:8000/ingest/image \
  -F 'title=sample uploaded image' \
  -F 'caption=uploaded image for similarity search' \
  -F 'file=@./samples/blue-mountain-lake.png' | jq
```

## Verify FAISS mapping

```bash
curl http://localhost:8000/debug/faiss-field | jq
```

Expected field config:

```json
{
  "field": "embedding",
  "type": "knn_vector",
  "dimension": 512,
  "method": {
    "name": "hnsw",
    "engine": "faiss",
    "space_type": "innerproduct",
    "parameters": {
      "m": 16,
      "ef_construction": 128
    }
  }
}
```

## Important implementation notes

### Why `innerproduct`?

The app normalizes every embedding vector. Once vectors are normalized, inner product behaves like cosine similarity.

### Where FAISS is configured

FAISS is configured in `app/main.py` inside the OpenSearch index mapping:

```python
"embedding": {
    "type": "knn_vector",
    "dimension": 512,
    "method": {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "innerproduct",
        "parameters": {
            "m": 16,
            "ef_construction": 128
        }
    }
}
```

### What OpenSearch stores

Each record stores:

```json
{
  "title": "...",
  "content": "...",
  "doc_type": "text | image | document",
  "source_path": "optional path for images",
  "metadata": {},
  "embedding": [0.01, -0.02, ...]
}
```

## Reset index

```bash
curl -X POST http://localhost:8000/admin/reset | jq
```
