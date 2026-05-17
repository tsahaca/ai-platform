# BGE Reranker + OpenSearch Zip-Ready Devcontainer Demo

This lab demonstrates a production-style retrieval flow:

```text
User Query
  -> OpenSearch BM25 lexical search
  -> OpenSearch k-NN vector search
  -> Hybrid candidate merge
  -> BGE cross-encoder reranker
  -> Top ranked chunks for RAG context
```

## What is included

- VS Code devcontainer
- Docker Compose for OpenSearch, OpenSearch Dashboards, and Python app container
- Sample document corpus
- OpenSearch BM25 retrieval
- OpenSearch k-NN vector retrieval using HNSW
- BGE reranker using `FlagEmbedding`
- CLI demo
- Streamlit UI

## Start

From the project root:

```bash
docker compose up -d --build
```

Or open the folder in VS Code and choose:

```text
Dev Containers: Reopen in Container
```

## Services

| Service | URL |
|---|---|
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| Streamlit UI | http://localhost:8501 |

Security is disabled for local development.

## Ingest sample documents

Inside the devcontainer:

```bash
python -m app.ingest
```

Expected output:

```text
Indexed 12 documents into bge_reranker_docs
```

## Run CLI demo

```bash
python -m app.main "heart attack symptoms"
python -m app.main "difference between BM25 and TF-IDF"
python -m app.main "best vector index for high recall"
python -m app.main "how reranking improves RAG"
```

## Run Streamlit UI

Inside the devcontainer:

```bash
streamlit run ui/streamlit_app.py --server.address 0.0.0.0
```

Open:

```text
http://localhost:8501
```

## How the demo works

### 1. BM25 retrieval

BM25 uses OpenSearch text fields:

```python
multi_match = {
    "query": query,
    "fields": ["title^2", "text"]
}
```

BM25 is keyword/lexical search. It does not require an LLM.

### 2. Vector retrieval

The demo creates embeddings with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Vectors are stored in OpenSearch as `knn_vector`:

```json
"embedding": {
  "type": "knn_vector",
  "dimension": 384,
  "method": {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "lucene"
  }
}
```

### 3. Hybrid merge

The app retrieves candidates from both BM25 and vector search, deduplicates by document id, and passes the candidates to the reranker.

### 4. BGE reranking

The reranker reads query/document pairs:

```python
pairs = [[query, candidate["text"]] for candidate in candidates]
scores = reranker.compute_score(pairs)
```

Default model:

```text
BAAI/bge-reranker-base
```

You can change it with:

```bash
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

## Recommended experiment queries

```text
heart attack symptoms
myocardial infarction emergency signs
what is BM25 search
how does HNSW improve vector search
fixed income duration spread risk
```

## Cleanup

```bash
docker compose down -v
```

## Notes

The first run downloads Hugging Face models, so it can take longer. The Docker volume `hf-cache` keeps models cached across container restarts.

## VS Code Dev Container Quick Start

1. Install the VS Code **Dev Containers** extension.
2. Open this folder in VS Code.
3. Choose **Dev Containers: Reopen in Container**.
4. The container starts these services automatically:
   - `opensearch` on port `9200`
   - `dashboards` on port `5601`
   - `app` dev shell
5. Run the included VS Code tasks in order:
   - `1. Wait for OpenSearch`
   - `2. Ingest sample documents`
   - `3. Run CLI demo`
   - `4. Start Streamlit UI`

Direct commands inside the devcontainer:

```bash
python -m app.ingest
python -m app.main "heart attack symptoms"
streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

OpenSearch Dashboards is available on forwarded port `5601`, and the Streamlit UI is available on forwarded port `8501`.
