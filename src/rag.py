# src/rag.py

from src.hybrid_search import HybridSearch
from src.llm import generate_text

class RAGPipeline:
    def __init__(self):
        self.search = HybridSearch(alpha=0.6)

    def build_context(self, docs, max_docs=5):
        context = "\n\n".join([d["document"]["text"] for d in docs[:max_docs]])
        return context

    def generate_answer(self, query):
        results = self.search.search(query, top_k=5)

        context = self.build_context(results)

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}

        Answer clearly and concisely.
        """

        answer = generate_text(prompt)

        return {
            "answer": answer,
            "documents": results
        }