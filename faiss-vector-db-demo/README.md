# Open Source Vector DB + FAISS Demo Stack

This project demonstrates five vector-search use cases using:

- **Qdrant** as the open-source vector database
- **FAISS** inside the FastAPI app as a local Approximate Nearest Neighbor search engine
- **SentenceTransformers** for text embeddings
- **FastAPI** for demo APIs

## Use cases covered

1. Semantic search: Google-like search for meaning
2. Recommendation systems
3. Chatbots / RAG pipelines
4. Image similarity search
5. Fraud detection: pattern similarity

## Architecture

```text
User / curl / Swagger UI
        |
        v
FastAPI app
        |
        |-- SentenceTransformers -> text embeddings
        |-- FAISS HNSW index -> fast local vector search
        |-- Qdrant -> persistent vector DB + metadata
        |
        v
JSON results
```

## Start the stack

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000/docs
```

Qdrant dashboard/API:

```text
http://localhost:6333/dashboard
```

## Smoke tests

### 1. Semantic search

```bash
curl "http://localhost:8000/semantic-search?q=how%20do%20I%20manage%20api%20traffic%20and%20policies%3F&engine=faiss" | jq
```

Compare with Qdrant:

```bash
curl "http://localhost:8000/semantic-search?q=how%20do%20I%20manage%20api%20traffic%20and%20policies%3F&engine=qdrant" | jq
```

### 2. Recommendation system

```bash
curl "http://localhost:8000/recommend?user_interest=I%20like%20hiking%20and%20mountain%20travel&engine=faiss" | jq
```

### 3. RAG chatbot pipeline

```bash
curl "http://localhost:8000/rag-chat?q=How%20should%20I%20debug%20curl%20exit%20code%2028%20in%20a%20gateway%20smoke%20test%3F&engine=faiss" | jq
```

This endpoint retrieves relevant context and returns a simple generated response. In production, send the retrieved context to Bedrock, Ollama, vLLM, or another LLM.

### 4. Image similarity search

```bash
curl "http://localhost:8000/image-similarity?image_id=red_circle&engine=faiss" | jq
```

Available image IDs:

```text
red_circle, red_square, blue_circle, blue_square, green_triangle, yellow_triangle
```

The demo uses lightweight image vectors based on color histograms and shape/edge hints. For production, replace `image_feature()` with CLIP embeddings.

### 5. Fraud pattern similarity

```bash
curl "http://localhost:8000/fraud-detect?amount=3000&hour=2&country_risk=0.9&velocity=8&chargeback_history=1&engine=faiss" | jq
```

Compare a normal transaction:

```bash
curl "http://localhost:8000/fraud-detect?amount=35&hour=14&country_risk=0.1&velocity=1&chargeback_history=0&engine=faiss" | jq
```

## FAISS vs Qdrant in this demo

| Component | Purpose |
|---|---|
| FAISS | Fast in-process vector similarity search |
| Qdrant | Persistent vector database with metadata, APIs, filtering, and storage |

In production, you usually choose one primary serving path:

- Use **Qdrant** directly when you want persistent vector DB features.
- Use **FAISS** directly when you want embedded/in-process vector search.
- Use both when you want to compare engines, build local indexes, or prototype retrieval behavior.

## Project structure

```text
faiss-vector-db-demo/
├── docker-compose.yml
├── README.md
└── app/
    ├── Dockerfile
    ├── requirements.txt
    └── main.py
```

## Notes

The first app startup downloads the SentenceTransformers model. This may take a few minutes depending on your network.

