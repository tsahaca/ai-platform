"""Simple LLM quality evaluator demo.

This project intentionally avoids paid APIs so it can run locally in a devcontainer.
It demonstrates three common LLM answer-quality dimensions:

1. Relevance: Does the answer address the question?
2. Factual accuracy: Does the answer agree with the reference/ground truth?
3. Fluency: Is the answer readable and well-formed?

The scoring logic is lightweight and educational, not a replacement for production
LLM-as-judge or human evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(str(text).lower()))


def cosine_tfidf(a: str, b: str) -> float:
    """Return TF-IDF cosine similarity between two strings."""
    a_norm, b_norm = normalize(a), normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([a_norm, b_norm])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class EvaluationResult:
    id: str
    relevance: float
    factual_accuracy: float
    fluency: float
    overall: float
    notes: str


def score_relevance(question: str, candidate_answer: str) -> float:
    """Approximate answer relevance using question-answer TF-IDF similarity."""
    return round(cosine_tfidf(question, candidate_answer), 3)


def score_factual_accuracy(reference_answer: str, candidate_answer: str) -> float:
    """Approximate factual accuracy using reference-candidate semantic overlap.

    A production implementation should compare claims against trusted evidence,
    use NLI/entailment models, human review, or LLM-as-judge with citations.
    """
    return round(cosine_tfidf(reference_answer, candidate_answer), 3)


def score_fluency(candidate_answer: str) -> float:
    """Approximate fluency using readability and basic grammar signals."""
    text = str(candidate_answer).strip()
    if not text:
        return 0.0

    words = _WORD_RE.findall(text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Reading ease is often 0-100+. Normalize around practical range.
    reading_ease = textstat.flesch_reading_ease(text)
    reading_score = clamp_0_1(reading_ease / 100.0)

    # Penalize extremely short fragments and missing punctuation for demo purposes.
    length_score = clamp_0_1(word_count / 12.0)
    punctuation_score = 1.0 if text[-1] in ".!?" else 0.7
    capital_score = 1.0 if text[0].isupper() else 0.8

    fluency = (0.45 * reading_score) + (0.25 * length_score) + (0.15 * punctuation_score) + (0.15 * capital_score)
    return round(clamp_0_1(fluency), 3)


def explain_result(relevance: float, factual: float, fluency: float) -> str:
    notes = []
    if relevance < 0.25:
        notes.append("Low relevance: candidate may not address the question.")
    if factual < 0.35:
        notes.append("Low factual accuracy: candidate does not strongly match reference answer.")
    if fluency < 0.55:
        notes.append("Low fluency: answer may be hard to read or incomplete.")
    return " ".join(notes) if notes else "Looks acceptable for this demo rubric."


def evaluate_row(row: pd.Series) -> EvaluationResult:
    relevance = score_relevance(row["question"], row["candidate_answer"])
    factual = score_factual_accuracy(row["reference_answer"], row["candidate_answer"])
    fluency = score_fluency(row["candidate_answer"])
    overall = round((0.3 * relevance) + (0.5 * factual) + (0.2 * fluency), 3)
    return EvaluationResult(
        id=str(row["id"]),
        relevance=relevance,
        factual_accuracy=factual,
        fluency=fluency,
        overall=overall,
        notes=explain_result(relevance, factual, fluency),
    )


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "question", "reference_answer", "candidate_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    results: Iterable[EvaluationResult] = (evaluate_row(row) for _, row in df.iterrows())
    return pd.DataFrame([asdict(result) for result in results])


def evaluate_csv(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    results = evaluate_dataframe(df)
    merged = pd.concat([df, results.drop(columns=["id"])], axis=1)
    if output_path:
        merged.to_csv(output_path, index=False)
    return merged
