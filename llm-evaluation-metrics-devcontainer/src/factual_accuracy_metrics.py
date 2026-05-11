from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def simple_groundedness_score(answer: str, context: str) -> float:
    """
    Simple demo metric: percentage of answer tokens found in context.
    Production systems usually use claim extraction + verifier model/LLM judge.
    """
    answer_tokens = set(normalize_text(answer).split())
    context_tokens = set(normalize_text(context).split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


if __name__ == "__main__":
    prediction = "Paris is the capital city of France."
    ground_truth = "Paris is the capital of France."
    context = "France has Paris as its capital. Paris is a major European city."

    print("Exact Match:", exact_match(prediction, ground_truth))
    print("Token F1:", token_f1(prediction, ground_truth))
    print("Groundedness:", simple_groundedness_score(prediction, context))
