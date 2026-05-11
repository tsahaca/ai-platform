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

## Diverse Test Dataset

This package now includes a richer dataset for testing relevance, factual accuracy, and fluency metrics:

```text
data/eval_dataset_diverse.jsonl
data/eval_dataset_diverse.json
data/eval_dataset_diverse.csv
```

It includes 15 scenarios:

- Perfect relevance, factuality, and fluency
- High relevance with hallucinated answer
- True but irrelevant answer
- Partial answer
- Unsupported claim where the model should abstain
- Ambiguous query
- Numeric near miss
- Date/temporal near miss
- Direct contradiction
- Verbose but correct answer
- Non-fluent but mostly correct answer
- Repetition-heavy answer
- Empty answer
- Complete retrieval failure
- Lexical-overlap trap with opposite meaning

Run the full dataset:

```bash
python src/evaluate_all.py
```

Run a specific dataset file:

```bash
python src/evaluate_all.py --dataset data/eval_dataset_diverse.jsonl --output evaluation_results_diverse.csv
```

### Dataset Fields

| Field | Meaning |
|---|---|
| `query` | User question |
| `relevant_docs` | Ground-truth relevant document IDs |
| `retrieved_docs` | Documents returned by the retriever |
| `ranking_relevance` | Graded relevance labels for NDCG |
| `ranking_scores` | Retriever/ranker scores |
| `source_context` | Context used for simple groundedness checks |
| `ground_truth` | Expected answer |
| `prediction` | Model answer to evaluate |
| `expected_*_quality` | Human-readable expectation for each metric family |
| `edge_case` | What the scenario is designed to test |
