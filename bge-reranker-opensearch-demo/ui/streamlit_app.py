import pandas as pd
import streamlit as st

from app.search import bm25_search, vector_search, hybrid_candidates, rerank

st.set_page_config(page_title="BGE Reranker + OpenSearch", layout="wide")
st.title("BGE Reranker + OpenSearch Demo")
st.caption("BM25 + OpenSearch k-NN retrieval, followed by BGE cross-encoder reranking")

query = st.text_input("Search query", value="heart attack symptoms")
k_each = st.slider("Candidates per retriever", 3, 20, 8)
top_k = st.slider("Reranked top K", 1, 10, 5)

if st.button("Search", type="primary"):
    bm25 = bm25_search(query, k_each)
    vector = vector_search(query, k_each)
    candidates = hybrid_candidates(query, k_each)
    reranked = rerank(query, candidates, top_k)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("BM25 lexical results")
        st.dataframe(pd.DataFrame(bm25)[["title", "category", "score", "text"]], use_container_width=True)
    with c2:
        st.subheader("Vector semantic results")
        st.dataframe(pd.DataFrame(vector)[["title", "category", "score", "text"]], use_container_width=True)

    st.subheader("BGE reranked hybrid results")
    st.dataframe(
        pd.DataFrame(reranked)[["title", "category", "reranker_score", "retrievers", "text"]],
        use_container_width=True,
    )
