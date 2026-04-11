from src.hybrid_search import HybridSearch
from src.quality_check import quality_score
from src.logger import log_failure
from src.feedback import analyze_failures


class RAGPipeline:
    def __init__(self):
        self.hybrid = HybridSearch(alpha=0.6)

    def generate_simple_answer(self, query, chunks):
        sentences = []

        for chunk in chunks:
            sentences += chunk.split(".")

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []

        for s in sentences:
            s_clean = s.strip().lower()

            if len(s_clean) < 40:
                continue

            words = set(s_clean.split())
            overlap_score = len(query_words.intersection(words))

            # 🔥 Definition query boost
            definition_bonus = 0
            if "what is" in query_lower:
                if " is " in s_clean or " refers to " in s_clean:
                    definition_bonus = 3

            final_score = overlap_score + definition_bonus

            scored.append((final_score, s.strip()))

        scored = sorted(scored, reverse=True)

        top_sentences = [s for _, s in scored[:2]]

        return ". ".join(top_sentences)

    def generate_answer(self, query):
        # 🔍 Retrieval
        results, confidence = self.hybrid.search(query, top_k=5, use_expansion=True)

        retrieved_chunks = [r["document"]["text"] for r in results]

        # 🤖 Generate answer
        answer = self.generate_simple_answer(query, retrieved_chunks)

        # 🔥 Query type detection
        query_lower = query.lower()
        query_type = "general"

        if any(w in query_lower for w in ["bill", "payment"]):
            query_type = "billing"
        elif "what is" in query_lower:
            query_type = "definition"

        # 📊 Quality + Faithfulness + Hallucination
        q_score, faith, halluc = quality_score(answer, retrieved_chunks, query_type)

        # 🔥 Trust score
        trust_score = 0.5 * confidence + 0.5 * q_score

        # 🔁 Feedback loop (dynamic threshold)
        feedback = analyze_failures()
        threshold = feedback["adjust_threshold"]

        # =========================================================
        # 🔥 NEW: VAGUE / NONSENSE QUERY DETECTION (SMART VERSION)
        # =========================================================

        query_words = query_lower.split()

        is_short = len(query_words) <= 2
        low_conf = confidence < 0.85
        low_quality = q_score < 0.6
        low_faith = faith < 0.5
        weak_signal = sum(len(w) for w in query_words) < 8

        if is_short and low_conf and (low_quality or low_faith) and weak_signal:
            log_failure({
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "quality": float(q_score),
                "faith": float(faith),
                "hallucination": float(halluc),
                "trust_score": float(trust_score),
                "reason": "vague_query"
            })

            return {
                "answer": "Query too vague or not meaningful enough to answer confidently.",
                "documents": results,
                "confidence": round(float(confidence), 3),
                "quality_score": round(float(q_score), 3),
                "faithfulness": round(float(faith), 3),
                "hallucination": 1.0,
                "trust_score": 0.0,
                "status": "ABSTAIN"
            }

        # =========================================================
        # 🔥 EXISTING ABSTENTION LOGIC
        # =========================================================

        # Primary: low trust
        if trust_score < threshold:
            log_failure({
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "quality": float(q_score),
                "faith": float(faith),
                "hallucination": float(halluc),
                "trust_score": float(trust_score),
                "reason": "low_trust"
            })

            return {
                "answer": "I'm not confident enough to answer. Please check documentation.",
                "documents": results,
                "confidence": round(float(confidence), 3),
                "quality_score": round(float(q_score), 3),
                "faithfulness": round(float(faith), 3),
                "hallucination": round(float(halluc), 3),
                "trust_score": round(float(trust_score), 3),
                "status": "ABSTAIN"
            }

        # Secondary: hallucination risk
        elif faith < 0.4 and halluc > 0.6:
            log_failure({
                "query": query,
                "answer": answer,
                "confidence": float(confidence),
                "quality": float(q_score),
                "faith": float(faith),
                "hallucination": float(halluc),
                "trust_score": float(trust_score),
                "reason": "hallucination"
            })

            return {
                "answer": "I'm not confident enough to answer. Please check documentation.",
                "documents": results,
                "confidence": round(float(confidence), 3),
                "quality_score": round(float(q_score), 3),
                "faithfulness": round(float(faith), 3),
                "hallucination": round(float(halluc), 3),
                "trust_score": round(float(trust_score), 3),
                "status": "ABSTAIN"
            }

        # =========================================================
        # ✅ SUCCESS
        # =========================================================

        return {
            "answer": answer,
            "documents": results,
            "confidence": round(float(confidence), 3),
            "quality_score": round(float(q_score), 3),
            "faithfulness": round(float(faith), 3),
            "hallucination": round(float(halluc), 3),
            "trust_score": round(float(trust_score), 3),
            "status": "PASS"
        }