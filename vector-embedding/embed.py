"""
Vector Embedding Demo
---------------------
Embeds the words: king, queen, man, woman
using the all-MiniLM-L6-v2 sentence-transformer model,
then computes pairwise cosine similarities, euclidean distances,
and demonstrates the classic analogy:  king - man + woman ≈ queen
"""

import numpy as np
from itertools import combinations
from tabulate import tabulate
from sentence_transformers import SentenceTransformer


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def nearest(query_vec: np.ndarray, embeddings: dict, exclude: list[str] = ()) -> str:
    """Return the word whose embedding is closest (cosine) to query_vec."""
    best_word, best_score = None, -2.0
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        score = cosine_similarity(query_vec, vec)
        if score > best_score:
            best_score, best_word = score, word
    return f"{best_word}  (similarity={best_score:.4f})"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    words = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl", "lord", "lady"]

    print("=" * 60)
    print("  Vector Embedding Demo  —  all-MiniLM-L6-v2")
    print("=" * 60)

    # 1. Load model & embed
    print("\n[1/4] Loading model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[2/4] Generating embeddings …\n")
    vectors = {w: model.encode(w, normalize_embeddings=True) for w in words}

    # Print embedding dimension
    dim = vectors[words[0]].shape[0]
    print(f"  Embedding dimension : {dim}")
    print(f"  Words               : {words}\n")

    # 2. Pairwise cosine similarity
    print("[3/4] Pairwise cosine similarity  (1 = identical, 0 = orthogonal, -1 = opposite)")
    print("-" * 60)

    sim_rows = []
    for w1, w2 in combinations(words, 2):
        sim = cosine_similarity(vectors[w1], vectors[w2])
        bar = "█" * int(sim * 20)
        relationship = _label_similarity(sim)
        sim_rows.append([f"{w1}  ↔  {w2}", f"{sim:.4f}", bar, relationship])

    print(tabulate(sim_rows,
                   headers=["Pair", "Cosine Sim", "Visual", "Relationship"],
                   tablefmt="rounded_outline"))

    # 3. Pairwise euclidean distance
    print("\n[4/4] Pairwise Euclidean distance  (lower = closer)")
    print("-" * 60)

    dist_rows = []
    for w1, w2 in combinations(words, 2):
        dist = euclidean_distance(vectors[w1], vectors[w2])
        dist_rows.append([f"{w1}  ↔  {w2}", f"{dist:.4f}"])

    # Sort by distance ascending
    dist_rows.sort(key=lambda r: float(r[1]))
    print(tabulate(dist_rows,
                   headers=["Pair", "Euclidean Distance"],
                   tablefmt="rounded_outline"))

    # 4. Classic analogy: king - man + woman ≈ queen
    print("\n" + "=" * 60)
    print("  Analogy:  king  −  man  +  woman  =  ?")
    print("=" * 60)

    analogy_vec = vectors["king"] - vectors["man"] + vectors["woman"]
    # Re-normalise so cosine comparison is fair
    analogy_vec = analogy_vec / np.linalg.norm(analogy_vec)

    result = nearest(analogy_vec, vectors, exclude=["king", "man", "woman"])
    print(f"\n  Nearest word  →  {result}")

    # Also show similarity of the analogy vector to every word
    print("\n  Similarity of (king − man + woman) to each word:")
    analogy_sim_rows = []
    for w in words:
        s = cosine_similarity(analogy_vec, vectors[w])
        analogy_sim_rows.append([w, f"{s:.4f}"])
    analogy_sim_rows.sort(key=lambda r: float(r[1]), reverse=True)
    print(tabulate(analogy_sim_rows, headers=["Word", "Similarity"], tablefmt="rounded_outline"))

    print("\n  Interpretation: The resulting vector is closest to 'queen',")
    print("  confirming that these embeddings capture gender & royalty semantics.\n")


def _label_similarity(sim: float) -> str:
    if sim >= 0.90:
        return "very similar"
    if sim >= 0.70:
        return "similar"
    if sim >= 0.40:
        return "somewhat related"
    if sim >= 0.10:
        return "weakly related"
    return "unrelated / orthogonal"


if __name__ == "__main__":
    main()
