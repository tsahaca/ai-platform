import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from app.evaluate import evaluate
from app.search_engines import BM25Search, HybridSearch, TfidfSearch, VectorSearch

st.set_page_config(page_title="Retrieval RAG Metrics Lab", layout="wide")
st.title("Retrieval RAG Metrics Lab")
st.write("Compare TF-IDF, BM25, vector search, hybrid search, and reranking.")

query = st.text_input("Query", "ADHD medication side effects")
k = st.slider("Top K", min_value=1, max_value=10, value=5)
engine_name = st.selectbox("Engine", ["tfidf", "bm25", "vector", "hybrid", "hybrid_rerank"])

engines = {
    "tfidf": TfidfSearch(),
    "bm25": BM25Search(),
    "vector": VectorSearch(),
    "hybrid": HybridSearch(),
    "hybrid_rerank": HybridSearch(),
}
engine = engines[engine_name]
results = engine.search(query, k=k, rerank=(engine_name == "hybrid_rerank")) if engine_name.startswith("hybrid") else engine.search(query, k=k)

st.subheader("Search results")
st.dataframe(pd.DataFrame(results)[["rank", "id", "title", "category", "score"] + (["rerank_score"] if "rerank_score" in results[0] else [])], use_container_width=True)

for r in results:
    with st.expander(f"{r['rank']}. {r['title']}"):
        st.write(r["text"])

st.subheader("Evaluation summary")
st.dataframe(evaluate(k=k), use_container_width=True)
