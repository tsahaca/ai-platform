from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from relevance_metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg
from factual_accuracy_metrics import exact_match, token_f1, simple_groundedness_score
from fluency_metrics import bleu, rouge, simple_fluency_heuristic

DEFAULT_DATASET_PATH = Path("data/eval_dataset_diverse.jsonl")


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def evaluate_item(item: dict) -> dict:
    rouge_scores = rouge(item["ground_truth"], item["prediction"])
    context = item.get("source_context", "") or (" ".join(item.get("relevant_docs", [])) + " " + item["ground_truth"])

    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "edge_case": item.get("edge_case"),
        "query": item["query"],
        "expected_relevance_quality": item.get("expected_relevance_quality"),
        "expected_factual_quality": item.get("expected_factual_quality"),
        "expected_fluency_quality": item.get("expected_fluency_quality"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relevance, factual accuracy, and fluency metrics.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to JSONL dataset")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = [evaluate_item(item) for item in load_dataset(Path(args.dataset))]
    df = pd.DataFrame(rows)

    display_columns = [
        "id", "category", "precision@3", "recall@3", "mrr", "ndcg",
        "exact_match", "token_f1", "groundedness_demo", "bleu", "rougeL_f1", "fluency_heuristic"
    ]
    print(df[display_columns].to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
