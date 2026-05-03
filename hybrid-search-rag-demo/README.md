# Hybrid Search + FAISS + OpenSearch + RAG Demo

This project shows all 3 next steps after TF-IDF and BM25:

1. A full hybrid search project with a devcontainer
2. FAISS and OpenSearch vector search side by side
3. A simple RAG pipeline using retrieved documents

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) FAISS semantic search

```bash
python -m app.faiss_search
```

## 2) BM25 + vector hybrid search

```bash
python -m app.tfidf_bm25_hybrid
```

## 3) Simple RAG demo

```bash
python -m app.rag_demo
```

## 4) Run FastAPI

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Then test:

```bash
curl "http://localhost:8000/search?q=solace%20tls%20hostname&k=3&alpha=0.5"
curl "http://localhost:8000/rag?q=How%20should%20pods%20connect%20to%20Solace%20with%20TLS%20SAN"
```

## 5) OpenSearch side-by-side demo

Start OpenSearch:

```bash
docker compose up -d opensearch
```

Run indexing and search:

```bash
python -m app.opensearch_demo
```

## Concepts

- BM25: lexical exact-word ranking
- Vector search: semantic similarity using embeddings
- Hybrid search: combines BM25 and vector scores
- RAG: retrieves relevant context, then generates an answer using that context
