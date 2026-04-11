# src/router.py

from src.intent_classifier import IntentClassifier
from src.tf_idf import TfidfRetriever
from src.hybrid_search import HybridSearch
from src.rag import RAGPipeline
from src.stats import StatsTracker
import time


class QueryRouter:
    def __init__(self):
        self.stats = StatsTracker()
        self.classifier = IntentClassifier()

        self.tfidf = TfidfRetriever()
        self.tfidf.load()

        self.hybrid = HybridSearch(alpha=0.6)
        self.rag = RAGPipeline()

    def route(self, query, top_k=5):
        start = time.time()

        intent, confidence = self.classifier.classify(query)

        # 🔥 Fallback condition
        if confidence < 0.75:
            results, _ = self.hybrid.search(query, top_k=top_k, use_expansion=True)
            latency = time.time() - start

            self.stats.log("fallback_hybrid", latency)

            return {
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "mode": "fallback_hybrid",
                "reason": "low intent confidence - fallback to hybrid search",
                "latency_ms": round(latency * 1000, 2),
                "results": results
            }

        # 🔹 TF-IDF (simple)
        if intent == "simple":
            results = self.tfidf.search(query, top_k=top_k)
            latency = time.time() - start

            self.stats.log("tfidf", latency)

            return {
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "mode": "tfidf",
                "reason": "simple query - lexical retrieval sufficient",
                "latency_ms": round(latency * 1000, 2),
                "results": results
            }

        # 🔹 Hybrid (semantic)
        elif intent == "semantic":
            results, _ = self.hybrid.search(query, top_k=top_k, use_expansion=True)
            latency = time.time() - start

            self.stats.log("hybrid", latency)

            return {
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "mode": "hybrid",
                "reason": "semantic query - hybrid retrieval used",
                "latency_ms": round(latency * 1000, 2),
                "results": results
            }

        # 🔹 RAG (complex)
        else:
            output = self.rag.generate_answer(query)
            latency = time.time() - start

            mode = "rag"
            reason = "complex query - requires generation"

            if output["status"] == "ABSTAIN":
                mode = "abstain"
                reason = "low confidence or low quality - avoided hallucination"

            self.stats.log(mode, latency)

            return {
                "query": query,
                "intent": intent,
                "confidence": confidence,
                "mode": mode,
                "reason": reason,
                "latency_ms": round(latency * 1000, 2),
                "answer": output["answer"],
                "results": output["documents"],
                "quality_score": output.get("quality_score"),
                "faithfulness": output.get("faithfulness"),
                "trust_score": output.get("trust_score"),
                "failure_type": output.get("failure_type")
            }