# Vector Recall Lab: FAISS vs OpenSearch HNSW

Zip-ready devcontainer project for learning production vector search.

## What this project demonstrates

- FAISS exact vector search as ground truth
- OpenSearch HNSW approximate vector search
- Recall@K comparison between FAISS and OpenSearch
- HNSW tuning:
  - `m`
  - `ef_construction`
  - `ef_search`
- BM25 search
- Vector search
- Hybrid search
- Streamlit visualization UI
- t-SNE embedding plot

## Architecture

```text
Sample Documents
   |
   v
SentenceTransformer Embeddings
   |
   +--> FAISS IndexFlatIP
   |       exact cosine baseline
   |
   +--> OpenSearch knn_vector
           HNSW approximate search
           BM25 + vector + hybrid
```

## Quick Start in VS Code Dev Containers

1. Unzip this project
2. Open folder in VS Code
3. Reopen in container
4. Wait for dependencies to install
5. Start OpenSearch:

```bash
docker compose up -d opensearch opensearch-dashboards
```

6. Load documents:

```bash
python scripts/load_opensearch.py
```

7. Start UI:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

OpenSearch Dashboards:

```text
http://localhost:5601
```

## Useful Commands

### Load / recreate OpenSearch index

```bash
python scripts/load_opensearch.py
```

### Search OpenSearch

```bash
python scripts/search_cli.py "cheap compute" --mode hybrid
python scripts/search_cli.py "cheap compute" --mode vector
python scripts/search_cli.py "cheap compute" --mode bm25
```

### Search FAISS

```bash
python scripts/faiss_demo.py "cheap compute"
```

### Compare recall

```bash
python scripts/compare_recall.py
```

### Tune HNSW

```bash
python scripts/tune_hnsw.py
```

## OpenSearch Dev Tools Queries

List indexes:

```json
GET _cat/indices?v
```

View documents:

```json
GET vector_lab_docs/_search
{
  "_source": ["id", "title", "category", "text"],
  "size": 20
}
```

Set ef_search dynamically:

```json
PUT vector_lab_docs/_settings
{
  "index.knn.algo_param.ef_search": 100
}
```

## Important Notes

FAISS `IndexFlatIP` with normalized embeddings is exact cosine similarity.

OpenSearch HNSW is approximate. It trades a tiny amount of recall for faster search at scale.

For small datasets, FAISS and OpenSearch may often return the same results. The value of this lab is learning the workflow and tuning controls.
