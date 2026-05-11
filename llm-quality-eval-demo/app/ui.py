from __future__ import annotations

import pandas as pd
import streamlit as st

from app.evaluator import evaluate_dataframe

st.set_page_config(page_title="LLM Quality Eval Demo", layout="wide")
st.title("LLM Quality Evaluation Demo")
st.write("Evaluate candidate LLM answers for relevance, factual accuracy, and fluency.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/eval_dataset.csv")

st.subheader("Input Dataset")
st.dataframe(df, use_container_width=True)

if st.button("Run Evaluation", type="primary"):
    results = evaluate_dataframe(df)
    merged = pd.concat([df, results.drop(columns=["id"])], axis=1)
    st.subheader("Evaluation Results")
    st.dataframe(merged, use_container_width=True)

    st.subheader("Average Scores")
    st.bar_chart(merged[["relevance", "factual_accuracy", "fluency", "overall"]].mean())

    st.download_button(
        "Download Results CSV",
        merged.to_csv(index=False),
        file_name="eval_results.csv",
        mime="text/csv",
    )
