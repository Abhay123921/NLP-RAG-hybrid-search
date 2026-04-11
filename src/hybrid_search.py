from src.tf_idf import TfidfRetriever
from src.embedding import EmbeddingModel
from src.faiss_index import FaissIndex
from src.query_expansion import expand_query
import numpy as np
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class HybridSearch:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

        self.tfidf = TfidfRetriever()
        self.tfidf.load()

        self.embedder = EmbeddingModel()

        self.faiss = FaissIndex(dim=384)
        self.faiss.load()

    def search(self, query, top_k=5, use_expansion=True):
        queries = [query]

        if use_expansion:
            queries = expand_query(query)

        tfidf_results = []
        bert_results = []

        for q in queries:
            tfidf_results += self.tfidf.search(q, top_k=top_k * 2)

            query_emb = self.embedder.encode_query(q)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            bert_results += self.faiss.search(query_emb, top_k=top_k * 2)

        tfidf_scores = {r["document"]["id"]: r["score"] for r in tfidf_results}
        bert_scores = {r["document"]["id"]: r["score"] for r in bert_results}

        all_ids = list(set(tfidf_scores.keys()) | set(bert_scores.keys()))

        tfidf_vals = np.array([tfidf_scores.get(i, 0) for i in all_ids])
        bert_vals = np.array([bert_scores.get(i, 0) for i in all_ids])

        tfidf_norm = (tfidf_vals - tfidf_vals.min()) / (tfidf_vals.max() - tfidf_vals.min() + 1e-8)
        bert_norm = (bert_vals - bert_vals.min()) / (bert_vals.max() - bert_vals.min() + 1e-8)

        combined = []

        for i, doc_id in enumerate(all_ids):
            final_score = self.alpha * tfidf_norm[i] + (1 - self.alpha) * bert_norm[i]

            doc = None
            for r in tfidf_results + bert_results:
                if r["document"]["id"] == doc_id:
                    doc = r["document"]
                    break

            combined.append({
                "score": float(final_score),
                "document": doc
            })

        # 🔥 Step 1: initial filtering
        combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:30]

        # 🔥 Step 2: reranking
        pairs = [(query, c["document"]["text"]) for c in combined]
        rerank_scores = reranker.predict(pairs)

        # ✅ FIX: Normalize reranker scores
        scores = np.array(rerank_scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        for i, c in enumerate(combined):
            c["score"] = float(scores[i])

        combined = sorted(combined, key=lambda x: x["score"], reverse=True)

        # 🔥 FIXED confidence (bounded)
        top_scores = np.array([r["score"] for r in combined[:top_k]])
        confidence = float(np.mean(top_scores))

        seen = set()
        unique = []

        for c in combined:
            text = c["document"]["text"]
            if text not in seen:
                seen.add(text)
                unique.append(c)

        combined = unique

        return combined[:top_k], confidence