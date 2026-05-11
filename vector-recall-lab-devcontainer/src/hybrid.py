def normalize_scores(results, score_field="score"):
    if not results:
        return []
    scores = [float(r.get(score_field, 0.0)) for r in results]
    min_s, max_s = min(scores), max(scores)
    output = []
    for r in results:
        item = r.copy()
        item["normalized_score"] = 1.0 if max_s == min_s else (float(item.get(score_field, 0.0)) - min_s) / (max_s - min_s)
        output.append(item)
    return output

def fuse_results(bm25_results, vector_results, alpha=0.5):
    bm25_norm = normalize_scores(bm25_results)
    vector_norm = normalize_scores(vector_results)
    combined = {}

    for r in bm25_norm:
        combined.setdefault(r["id"], r.copy())
        combined[r["id"]]["bm25_score_norm"] = r["normalized_score"]
        combined[r["id"]].setdefault("vector_score_norm", 0.0)

    for r in vector_norm:
        combined.setdefault(r["id"], r.copy())
        combined[r["id"]]["vector_score_norm"] = r["normalized_score"]
        combined[r["id"]].setdefault("bm25_score_norm", 0.0)

    fused = []
    for r in combined.values():
        r["hybrid_score"] = alpha * r.get("bm25_score_norm", 0.0) + (1 - alpha) * r.get("vector_score_norm", 0.0)
        fused.append(r)

    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    for i, r in enumerate(fused):
        r["rank"] = i + 1
    return fused
