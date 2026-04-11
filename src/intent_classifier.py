class IntentClassifier:
    def __init__(self):
        pass

    def classify(self, query: str):
        query_lower = query.lower()
        words = query_lower.split()

        # 🔥 1. Question → complex
        if any(w in query_lower for w in ["what", "how", "why", "explain", "define"]):
            return "complex", 0.85

        # 🔥 2. Multi-keyword technical queries → semantic
        if len(words) >= 2:
            return "semantic", 0.8

        # 🔥 3. Very short → simple
        if len(words) <= 1:
            return "simple", 0.9

        return "semantic", 0.7