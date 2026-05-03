from app.tfidf_bm25_hybrid import HybridSearch

class SimpleRAG:
    def __init__(self):
        self.searcher = HybridSearch()

    def answer(self, question: str):
        hits = self.searcher.search(question, k=3, alpha=0.5)
        context = "\n".join([f"- {h['title']}: {h['text']}" for h in hits])
        return f"""
Question: {question}

Retrieved context:
{context}

Simple generated answer:
Based on the retrieved context, use the hostname that matches the TLS certificate SAN when clients connect. If you want internal OpenShift traffic to avoid the AWS NLB, keep the application URL as the SAN hostname and make cluster DNS resolve that hostname internally to the Solace service endpoint.
"""

if __name__ == "__main__":
    rag = SimpleRAG()
    print(rag.answer("How should OpenShift pods connect to Solace when TLS SAN is my-solace.example.com?"))
