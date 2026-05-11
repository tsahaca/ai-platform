from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from relevance_metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg
from factual_accuracy_metrics import exact_match, token_f1, simple_groundedness_score
from fluency_metrics import bleu, rouge, simple_fluency_heuristic

DATASET_PATH = Path("data/eval_dataset.jsonl")


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    rows = []
    for item in load_dataset(DATASET_PATH):
        rouge_scores = rouge(item["ground_truth"], item["prediction"])
        context = " ".join(item.get("relevant_docs", [])) + " " + item["ground_truth"]

        rows.append(
            {
                "query": item["query"],
                "precision@3": precision_at_k(item["relevant_docs"], item["retrieved_docs"], 3),
                "recall@3": recall_at_k(item["relevant_docs"], item["retrieved_docs"], 3),
                "mrr": reciprocal_rank(item["relevant_docs"], item["retrieved_docs"]),
                "ndcg": ndcg(item["ranking_relevance"], item["ranking_scores"]),
                "exact_match": exact_match(item["prediction"], item["ground_truth"]),
                "token_f1": token_f1(item["prediction"], item["ground_truth"]),
                "groundedness_demo": simple_groundedness_score(item["prediction"], context),
                "bleu": bleu(item["ground_truth"], item["prediction"]),
                "rougeL_f1": rouge_scores["rougeL"]["fmeasure"],
                "fluency_heuristic": simple_fluency_heuristic(item["prediction"]),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved: evaluation_results.csv")


if __name__ == "__main__":
    main()
