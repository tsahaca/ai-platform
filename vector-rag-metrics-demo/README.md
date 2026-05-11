# Vector Search / RAG Metrics Demo

Zip-ready VS Code Dev Container project to demo these RAG retrieval metrics:

- `Recall@K`
- `MRR`
- `nDCG`

The project uses a small local corpus, example RAG-style queries, graded relevance labels, and a lightweight TF-IDF vector retriever. You can replace the retriever later with FAISS, OpenSearch k-NN, Bedrock Knowledge Bases, or another vector database.

## Project Structure

```text
.
├── .devcontainer/devcontainer.json
├── app/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── retriever.py
│   └── ui.py
├── data/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels.jsonl
├── tests/test_metrics.py
├── requirements.txt
└── README.md
```

## Open in Dev Container

1. Unzip this package.
2. Open the folder in VS Code.
3. Choose **Reopen in Container**.
4. The container installs dependencies automatically.

## Run CLI Evaluation

```bash
python -m app.evaluate --k 5
```

Example output:

```text
Recall@5
MRR@5
nDCG@5
```

## Run Streamlit UI

```bash
streamlit run app/ui.py --server.address 0.0.0.0 --server.port 8501
```

Then open the forwarded port `8501`.

## Run Tests

```bash
pytest -q
```

## Concepts

### Recall@K

Measures how many known relevant documents were retrieved in the top K.

```text
Recall@K = relevant documents retrieved in top K / total relevant documents
```

Use this when you care about whether the retriever found enough useful context for RAG.

### MRR

Measures how early the first relevant document appears.

```text
MRR = average(1 / rank of first relevant result)
```

Use this when the first good result matters most.

### nDCG

Measures ranking quality with graded relevance and position discounting.

```text
nDCG@K = DCG@K / Ideal DCG@K
```

Use this when some documents are more relevant than others and higher-ranked relevant documents should be rewarded more.

## How to Extend

To plug in a real vector search backend:

1. Replace `app/retriever.py` with FAISS, OpenSearch, or Bedrock Knowledge Bases retrieval.
2. Keep the return shape:

```python
[
  {"rank": 1, "doc_id": "doc_id", "score": 0.95, "text": "..."}
]
```

3. Keep `data/qrels.jsonl` as your ground truth relevance set.
4. Run the same metrics unchanged.
