from __future__ import annotations

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def bleu(reference: str, candidate: str) -> float:
    reference_tokens = [reference.lower().split()]
    candidate_tokens = candidate.lower().split()
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing))


def rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        metric: {
            "precision": score.precision,
            "recall": score.recall,
            "fmeasure": score.fmeasure,
        }
        for metric, score in scores.items()
    }


def simple_fluency_heuristic(text: str) -> float:
    """
    Demo-only fluency heuristic.
    Gives a small penalty for very short, fragment-like answers and repeated words.
    """
    tokens = text.lower().split()
    if not tokens:
        return 0.0

    repeated_pairs = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    length_score = min(len(tokens) / 8, 1.0)
    repetition_penalty = repeated_pairs / max(len(tokens), 1)
    return max(0.0, length_score - repetition_penalty)


if __name__ == "__main__":
    reference = "Paris is the capital city of France."
    candidate = "Paris is the capital of France."

    print("BLEU:", bleu(reference, candidate))
    print("ROUGE:", rouge(reference, candidate))
    print("Simple Fluency Heuristic:", simple_fluency_heuristic(candidate))
