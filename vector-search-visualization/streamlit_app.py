import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = [
"AWS Lambda is serverless compute",
"EC2 provides virtual machines",
"S3 is object storage",
"Azure Functions is serverless",
"Google Cloud Run runs containers",
"Kubernetes orchestrates containers",
"Cheap cloud compute options",
"Low cost serverless architecture",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, normalize_embeddings=True).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

st.title("Vector Search Demo")

query = st.text_input("Enter query")

if query:
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, 3)
    for i in ids[0]:
        st.write(docs[i])
