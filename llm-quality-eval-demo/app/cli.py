from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from app.evaluator import evaluate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo evaluator for LLM answer quality")
    parser.add_argument("--input", default="data/eval_dataset.csv", help="CSV with id, question, reference_answer, candidate_answer")
    parser.add_argument("--output", default="outputs/eval_results.csv", help="Output CSV path")
    args = parser.parse_args()

    results = evaluate_csv(args.input, args.output)

    table = Table(title="LLM Quality Evaluation Results")
    for col in ["id", "relevance", "factual_accuracy", "fluency", "overall", "notes"]:
        table.add_column(col)

    for _, row in results.iterrows():
        table.add_row(
            str(row["id"]),
            f"{row['relevance']:.3f}",
            f"{row['factual_accuracy']:.3f}",
            f"{row['fluency']:.3f}",
            f"{row['overall']:.3f}",
            str(row["notes"]),
        )

    console = Console()
    console.print(table)
    console.print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
