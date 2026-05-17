# Retrieval RAG Metrics Lab

Zip-ready devcontainer lab for learning:

- TF-IDF vs BM25
- Sparse lexical search vs dense vector search
- FAISS cosine similarity search
- Hybrid retrieval
- Simple educational reranking
- Recall@K, MRR, and nDCG
- Optional OpenSearch + OpenSearch Dashboards service

## Open in VS Code Dev Containers

1. Install Docker Desktop.
2. Install the VS Code Dev Containers extension.
3. Open this folder in VS Code.
4. Choose **Reopen in Container**.

## Run the local demos

```bash
python -m app.demo
python -m app.evaluate
```

## Run Streamlit UI

```bash
streamlit run app/ui.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Run JupyterLab

```bash
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
```

Open the URL printed in the terminal.

## Load sample docs into OpenSearch

The compose file starts OpenSearch on port `9200` and Dashboards on `5601`.

```bash
python -m app.opensearch_load
```

Then open:

```text
http://localhost:5601
```

Create a data view for index:

```text
retrieval_lab_docs
```

## What to observe

Try these queries:

```text
ADHD medication side effects
medicine for focus disorder
AWS iam:PassRole condition
how does OpenSearch rank keyword documents
HNSW vector similarity search
reranking candidates in RAG
validate SOAP payload using XSD
```

### Key learning points

- **TF-IDF** is sparse vector lexical retrieval. It can over-reward repeated terms.
- **BM25** improves TF-IDF-style ranking with term saturation and document-length normalization.
- **Vector search** uses dense embeddings and cosine similarity to capture semantic meaning.
- **Hybrid search** combines BM25 exact-token strength with embedding semantic matching.
- **Reranking** performs a second pass over retrieved candidates to improve final order.
- **Recall@K** measures whether relevant docs appear in the top K.
- **MRR** rewards the first relevant result appearing early.
- **nDCG** rewards highly relevant docs near the top of the ranking.

## Project layout

```text
.
├── .devcontainer/devcontainer.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app
│   ├── data.py
│   ├── demo.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── opensearch_load.py
│   ├── search_engines.py
│   └── ui.py
└── data
    ├── docs.jsonl
    └── queries.jsonl
```
