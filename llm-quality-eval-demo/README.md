# LLM Quality Evaluation Demo

Zip-ready VS Code Dev Container project to demonstrate three LLM answer-quality metrics:

1. **Relevance** — does the answer address the question?
2. **Factual Accuracy** — does the answer agree with the reference answer / ground truth?
3. **Fluency** — is the answer readable, grammatical, and natural?

This demo runs locally and does **not** require OpenAI, Bedrock, or any paid LLM API.

---

## Project Structure

```text
llm-quality-eval-demo/
├── .devcontainer/
│   └── devcontainer.json
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── evaluator.py
│   └── ui.py
├── data/
│   └── eval_dataset.csv
├── tests/
│   └── test_evaluator.py
├── Makefile
├── README.md
└── requirements.txt
```

---

## Metric Logic

### 1. Relevance

The demo compares the **question** and **candidate answer** using TF-IDF cosine similarity.

High relevance example:

```text
Question: What is RAG?
Answer: RAG combines retrieval of external knowledge with LLM generation.
```

Low relevance example:

```text
Question: What is RAG?
Answer: OpenSearch is used for log analytics.
```

---

### 2. Factual Accuracy

The demo compares the **reference answer** and **candidate answer** using TF-IDF cosine similarity.

High factual accuracy example:

```text
Reference: Paris is the capital of France.
Candidate: Paris is the capital city of France.
```

Low factual accuracy example:

```text
Reference: Paris is the capital of France.
Candidate: Lyon is the capital of France.
```

In production, factual accuracy should be evaluated with stronger methods, such as:

- human review
- LLM-as-judge
- citation/evidence checking
- entailment/NLI models
- claim extraction + verification

---

### 3. Fluency

The demo uses simple readability signals:

- Flesch reading ease
- answer length
- punctuation
- capitalization

High fluency example:

```text
Amazon S3 is an object storage service used to store and retrieve data at scale.
```

Low fluency example:

```text
S3 object storage scale data yes maybe retrieve
```

---

## Run in VS Code Dev Container

1. Unzip the project.
2. Open the folder in VS Code.
3. Install the **Dev Containers** extension if needed.
4. Select:

```text
Dev Containers: Reopen in Container
```

The container installs dependencies automatically.

---

## Run CLI Demo

```bash
make run
```

Or directly:

```bash
mkdir -p outputs
python -m app.cli --input data/eval_dataset.csv --output outputs/eval_results.csv
```

Expected output:

```text
LLM Quality Evaluation Results
id | relevance | factual_accuracy | fluency | overall | notes
```

Results are saved to:

```text
outputs/eval_results.csv
```

---

## Run Streamlit UI

```bash
make ui
```

Then open:

```text
http://localhost:8501
```

The UI lets you:

- view the sample dataset
- upload your own CSV
- run evaluation
- see score table
- view average score chart
- download results

---

## CSV Format

Your CSV must contain these columns:

```csv
id,question,reference_answer,candidate_answer
```

Example:

```csv
1,What is RAG?,RAG combines retrieval with generation.,RAG uses search results to help an LLM answer.
```

---

## Run Tests

```bash
make test
```

---

## Important Note

This is an educational scoring demo. TF-IDF similarity can miss many factual errors.

For production LLM evaluation, use this pattern:

```text
Dataset → Candidate Answer → Evaluator → Scores → Human Review / CI Quality Gate
```

Recommended production metrics:

- Relevance
- Factual Accuracy / Groundedness
- Fluency
- Toxicity / Safety
- Citation correctness
- Retrieval Recall@K
- Answer latency
- Cost per answer
