import streamlit as st
import pandas as pd
from app.data_loader import load_corpus, load_queries, load_qrels
from app.retriever import TfidfVectorRetriever
from app.metrics import evaluate_query
from app.evaluate import run_evaluation

st.set_page_config(page_title="RAG Metrics Demo", layout="wide")
st.title("Vector Search / RAG Metrics Demo")
st.write("Demo metrics: Recall@K, MRR, and nDCG using a small corpus and ground-truth relevance labels.")

k = st.slider("Top K", min_value=1, max_value=10, value=5)
corpus = load_corpus()
queries = load_queries()
qrels = load_qrels()
retriever = TfidfVectorRetriever(corpus)

st.header("Full Evaluation")
df, summary, _ = run_evaluation(k=k)
st.dataframe(df, use_container_width=True)
st.write("Mean metrics")
st.json({name: round(value, 4) for name, value in summary.items()})

st.header("Inspect One Query")
query_labels = {f'{q["query_id"]}: {q["query"]}': q for q in queries}
selected_label = st.selectbox("Select query", list(query_labels.keys()))
selected = query_labels[selected_label]
results = retriever.search(selected["query"], k=k)
ranked_doc_ids = [r["doc_id"] for r in results]
metrics = evaluate_query(ranked_doc_ids, qrels[selected["query_id"]], k=k)

left, right = st.columns([1, 2])
with left:
    st.subheader("Metrics")
    st.json({name: round(value, 4) for name, value in metrics.items()})
    st.subheader("Ground Truth Relevance")
    st.dataframe(pd.DataFrame([
        {"doc_id": doc_id, "relevance": rel}
        for doc_id, rel in qrels[selected["query_id"]].items()
    ]), use_container_width=True)
with right:
    st.subheader("Retrieved Results")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

st.header("Metric Meaning")
st.markdown("""
- **Recall@K**: fraction of all relevant documents found in the top K.
- **MRR@K**: reciprocal rank of the first relevant document in the top K.
- **nDCG@K**: ranking quality using graded relevance and lower weight for lower-ranked results.
""")
