import pandas as pd

from app.evaluator import evaluate_dataframe, score_fluency, score_factual_accuracy


def test_factual_similarity_higher_for_correct_answer():
    reference = "Paris is the capital of France."
    good = "Paris is the capital of France."
    bad = "Lyon is the capital of France."
    assert score_factual_accuracy(reference, good) >= score_factual_accuracy(reference, bad)


def test_fluency_non_empty_answer_scores_above_zero():
    assert score_fluency("This is a clear and readable sentence.") > 0


def test_evaluate_dataframe_returns_expected_columns():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "question": "What is RAG?",
                "reference_answer": "RAG combines retrieval with generation.",
                "candidate_answer": "RAG combines retrieval with generated answers.",
            }
        ]
    )
    results = evaluate_dataframe(df)
    assert {"relevance", "factual_accuracy", "fluency", "overall", "notes"}.issubset(results.columns)
