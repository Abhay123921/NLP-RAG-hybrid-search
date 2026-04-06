from src.tf_idf import TfidfRetriever
from src.embedding import EmbeddingModel
from src.faiss_index import FaissIndex
from src.utils import normalize
from src.query_expansion import expand_query
import numpy as np


class HybridSearch:
    def __init__(self, alpha=0.2):
        self.alpha = alpha  # balance factor

        self.tfidf = TfidfRetriever()
        self.tfidf.load()

        self.embedder = EmbeddingModel()

        self.faiss = FaissIndex(dim=384)
        self.faiss.load()

    def normalize_scores(self, scores_dict):
        scores = np.array(list(scores_dict.values()))
        min_s, max_s = scores.min(), scores.max()
    
        normalized = {}
        for k, v in scores_dict.items():
            normalized[k] = (v - min_s) / (max_s - min_s + 1e-8)
    
        return normalized

    def search(self, query, top_k=5, use_expansion=True):

        queries = [query]

        if use_expansion:
            queries = expand_query(query)

        all_results = []

        for q in queries:
            tfidf_results = self.tfidf.search(q, top_k=top_k*2)

            query_emb = self.embedder.encode_query(q)
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

            bert_results = self.faiss.search(query_emb, top_k=top_k*2)

            tfidf_scores = {r["document"]["id"]: r["score"] for r in tfidf_results}
            bert_scores = {r["document"]["id"]: r["score"] for r in bert_results}

            all_ids = list(set(tfidf_scores.keys()) | set(bert_scores.keys()))

            tfidf_vals = np.array([tfidf_scores.get(i, 0) for i in all_ids])
            bert_vals = np.array([bert_scores.get(i, 0) for i in all_ids])

            tfidf_norm = (tfidf_vals - tfidf_vals.min()) / (tfidf_vals.max() - tfidf_vals.min() + 1e-8)
            bert_norm = (bert_vals - bert_vals.min()) / (bert_vals.max() - bert_vals.min() + 1e-8)

            for i, doc_id in enumerate(all_ids):
                final_score = self.alpha * tfidf_norm[i] + (1 - self.alpha) * bert_norm[i]

                doc = None
                for r in tfidf_results + bert_results:
                    if r["document"]["id"] == doc_id:
                        doc = r["document"]
                        break

                all_results.append({
                    "score": float(final_score),
                    "document": doc
                })

        # 🔥 Deduplicate + rerank
        unique = {}
        for r in all_results:
            doc_id = r["document"]["id"]
            if doc_id not in unique or r["score"] > unique[doc_id]["score"]:
                unique[doc_id] = r

        final_results = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

        return final_results[:top_k]