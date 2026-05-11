import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE
import plotly.express as px

from src.data import DOCUMENTS, EVAL_QUERIES
from src.embeddings import embed_texts
from src.faiss_store import FaissExactStore
from src.opensearch_store import (
    bm25_search,
    vector_search,
    hybrid_search,
    create_index,
    load_documents,
    update_ef_search,
    get_client,
)
from src.metrics import recall_at_k, timer

st.set_page_config(page_title="Vector Recall Lab", layout="wide")

st.title("Vector Recall Lab: FAISS vs OpenSearch HNSW")
st.caption("Visualize vector search, compare exact FAISS with OpenSearch HNSW, and tune recall/latency.")

@st.cache_resource
def get_faiss():
    return FaissExactStore()

@st.cache_data
def get_embedding_plot():
    texts = [d["title"] + " " + d["text"] for d in DOCUMENTS]
    vectors = embed_texts(texts, normalize=True)
    perplexity = min(5, len(DOCUMENTS) - 1)
    reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(vectors)
    return pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "id": [d["id"] for d in DOCUMENTS],
        "title": [d["title"] for d in DOCUMENTS],
        "category": [d["category"] for d in DOCUMENTS],
        "text": [d["text"] for d in DOCUMENTS],
    })

faiss_store = get_faiss()

with st.sidebar:
    st.header("OpenSearch HNSW Controls")
    m = st.slider("m", 4, 48, 16, 4)
    ef_construction = st.slider("ef_construction", 32, 512, 128, 32)
    ef_search = st.slider("ef_search", 10, 300, 100, 10)

    if st.button("Recreate OpenSearch Index"):
        try:
            client = get_client()
            create_index(client, recreate=True, m=m, ef_construction=ef_construction, ef_search=ef_search)
            load_documents(client)
            st.success("Index recreated and documents loaded.")
        except Exception as e:
            st.error(str(e))

    if st.button("Update ef_search Only"):
        try:
            update_ef_search(ef_search)
            st.success("ef_search updated.")
        except Exception as e:
            st.error(str(e))

tab1, tab2, tab3, tab4 = st.tabs(["Search Compare", "Recall Evaluation", "Embedding Map", "OpenSearch Queries"])

with tab1:
    query = st.text_input("Query", value="cheap compute")
    k = st.slider("Top K", 1, 10, 5)
    alpha = st.slider("Hybrid BM25 Weight", 0.0, 1.0, 0.5, 0.1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("FAISS Exact Baseline")
        with timer() as t:
            faiss_results = faiss_store.search(query, k=k)
        st.caption(f"Latency: {t['elapsed_ms']} ms")
        st.dataframe(pd.DataFrame(faiss_results)[["rank", "id", "title", "category", "score"]], use_container_width=True)

    with col2:
        st.subheader("OpenSearch Vector HNSW")
        try:
            with timer() as t:
                os_results = vector_search(query, k=k)
            st.caption(f"Latency: {t['elapsed_ms']} ms")
            st.dataframe(pd.DataFrame(os_results)[["id", "title", "category", "score"]], use_container_width=True)
            recall = recall_at_k([r["id"] for r in faiss_results], [r["id"] for r in os_results], k)
            st.metric(f"Recall@{k} vs FAISS", f"{recall:.2f}")
        except Exception as e:
            st.error(str(e))
            st.info("Start OpenSearch and run: python scripts/load_opensearch.py")

    st.subheader("BM25 and Hybrid")
    c1, c2 = st.columns(2)
    with c1:
        try:
            bm25_results = bm25_search(query, k=k)
            st.markdown("**BM25**")
            st.dataframe(pd.DataFrame(bm25_results)[["id", "title", "category", "score"]], use_container_width=True)
        except Exception as e:
            st.warning(str(e))
    with c2:
        try:
            hybrid_results = hybrid_search(query, k=k, alpha=alpha)
            st.markdown("**Hybrid**")
            st.dataframe(pd.DataFrame(hybrid_results)[["rank", "id", "title", "category", "hybrid_score"]], use_container_width=True)
        except Exception as e:
            st.warning(str(e))

with tab2:
    st.subheader("Recall@K Evaluation")
    eval_k = st.slider("Evaluation K", 1, 10, 5)
    rows = []
    try:
        for item in EVAL_QUERIES:
            q = item["query"]
            with timer() as tf:
                f_ids = faiss_store.search_ids(q, k=eval_k)
            with timer() as to:
                o_ids = [r["id"] for r in vector_search(q, k=eval_k)]
            rows.append({
                "query": q,
                "faiss_top_ids": ", ".join(f_ids),
                "opensearch_top_ids": ", ".join(o_ids),
                "recall_vs_faiss": recall_at_k(f_ids, o_ids, eval_k),
                "faiss_ms": tf["elapsed_ms"],
                "opensearch_ms": to["elapsed_ms"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.metric(f"Average Recall@{eval_k}", f"{df['recall_vs_faiss'].mean():.2f}")
        st.bar_chart(df.set_index("query")["recall_vs_faiss"])
    except Exception as e:
        st.error(str(e))

with tab3:
    st.subheader("Embedding Map")
    df_plot = get_embedding_plot()
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="category",
        hover_data=["id", "title", "text"],
        text="title",
        title="t-SNE projection of document embeddings"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("OpenSearch Dashboard / Dev Tools")
    st.markdown("""
Open OpenSearch Dashboards:

```text
http://localhost:5601
```

Useful Dev Tools queries:

```json
GET _cat/indices?v
```

```json
GET vector_lab_docs/_search
{
  "_source": ["id", "title", "category", "text"],
  "size": 20
}
```

```json
GET vector_lab_docs/_mapping
```

```json
PUT vector_lab_docs/_settings
{
  "index.knn.algo_param.ef_search": 100
}
```
""")
