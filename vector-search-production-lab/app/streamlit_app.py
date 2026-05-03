import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from src.faiss_store import FaissVectorStore
from src.opensearch_store import bm25_search, vector_search, hybrid_search

st.set_page_config(page_title="Vector Search Production Lab", layout="wide")
st.title("Vector Search Production Lab")

mode = st.sidebar.selectbox("Search Mode", ["hybrid", "bm25", "vector", "faiss"])
k = st.sidebar.slider("Top K", 1, 10, 5)
alpha = st.sidebar.slider("Hybrid BM25 weight", 0.0, 1.0, 0.5, 0.1)

query = st.text_input("Search query", value="cheap compute")

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = FaissVectorStore()

if query:
    try:
        if mode == "bm25":
            results = bm25_search(query, k=k)
        elif mode == "vector":
            results = vector_search(query, k=k)
        elif mode == "hybrid":
            results = hybrid_search(query, k=k, alpha=alpha)
        else:
            results = st.session_state.faiss_store.search(query, k=k)

        rows = [{
            "rank": r.get("rank"),
            "title": r["title"],
            "category": r["category"],
            "score": r.get("hybrid_score", r.get("score")),
            "text": r["text"]
        } for r in results]

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    except Exception as e:
        st.error(str(e))
        st.info("For OpenSearch modes, run: docker compose up -d opensearch && python scripts/load_opensearch.py")
