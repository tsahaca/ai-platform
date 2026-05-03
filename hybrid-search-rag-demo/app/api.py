from fastapi import FastAPI, Query
from app.tfidf_bm25_hybrid import HybridSearch
from app.rag_demo import SimpleRAG

app = FastAPI(title="Hybrid Search + RAG Demo")
searcher = HybridSearch()
rag = SimpleRAG()

@app.get("/search")
def search(q: str = Query(...), k: int = 3, alpha: float = 0.5):
    return {"query": q, "results": searcher.search(q, k=k, alpha=alpha)}

@app.get("/rag")
def rag_answer(q: str = Query(...)):
    return {"answer": rag.answer(q)}
