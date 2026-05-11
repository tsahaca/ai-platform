# LLM Evaluation Metrics Lab

A zip-ready VS Code Dev Container project for learning and running evaluation metrics for:

1. Relevance
2. Factual accuracy
3. Fluency

## Project Structure

```text
.
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── data/
│   └── eval_dataset.jsonl
├── src/
│   ├── relevance_metrics.py
│   ├── factual_accuracy_metrics.py
│   ├── fluency_metrics.py
│   └── evaluate_all.py
├── requirements.txt
└── README.md
```

## Open in VS Code Dev Containers

1. Install Docker Desktop.
2. Install VS Code.
3. Install the **Dev Containers** extension.
4. Open this folder in VS Code.
5. Choose **Reopen in Container**.

## Run Individual Examples

```bash
python src/relevance_metrics.py
python src/factual_accuracy_metrics.py
python src/fluency_metrics.py
```

## Run Full Evaluation

```bash
python src/evaluate_all.py
```

This generates:

```text
evaluation_results.csv
```

## Metrics Included

### Relevance

- Precision@K
- Recall@K
- MRR
- NDCG

### Factual Accuracy

- Exact Match
- Token F1
- Simple groundedness demo score

### Fluency

- BLEU
- ROUGE
- Simple fluency heuristic

## Notes

The groundedness and fluency heuristic examples are intentionally simple so the math is easy to understand. In production RAG systems, factual accuracy is often evaluated with claim extraction, source attribution, and LLM-as-a-judge methods.
