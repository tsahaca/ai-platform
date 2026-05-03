# Vector Search Production Lab

Hands-on production-style project for learning:

- FAISS local vector search
- OpenSearch `knn_vector`
- BM25 lexical search
- Hybrid search: BM25 + vector
- Streamlit UI
- Latency/relevance evaluation
- OpenSearch Dashboards visualization

## Run OpenSearch

```bash
docker compose up -d opensearch opensearch-dashboards
```

OpenSearch:

```text
http://localhost:9200
```

OpenSearch Dashboards:

```text
http://localhost:5601
```

## Run In VS Code Dev Container

This repository includes a dev container config in `.devcontainer/`.

In VS Code:

1. Open the repository.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for `postCreateCommand` to finish installing dependencies.

Inside the container, run:

```bash
python scripts/load_opensearch.py
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Ports forwarded by default:

```text
8501 (Streamlit), 9200 (OpenSearch), 5601 (Dashboards)
```

## Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Load sample documents

```bash
python scripts/load_opensearch.py
```

## Search from CLI

```bash
python scripts/search_cli.py "cheap compute" --mode hybrid
python scripts/search_cli.py "cheap compute" --mode bm25
python scripts/search_cli.py "cheap compute" --mode vector
python scripts/faiss_demo.py "cheap compute"
```

## Run UI

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Key Concepts

```text
Inverted Index / BM25:
word -> document list

Vector Index:
embedding -> nearest vector neighbors

Hybrid Search:
BM25 score + vector similarity score
```

## Important Files

```text
opensearch/index_mapping.json     OpenSearch schema with knn_vector
src/faiss_store.py                FAISS cosine similarity search
src/opensearch_store.py           OpenSearch BM25/vector/hybrid search
src/hybrid.py                     Score fusion logic
app/streamlit_app.py              UI
scripts/evaluate.py               Simple evaluation
notebooks/vector_visualization.ipynb
```
