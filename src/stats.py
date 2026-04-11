class StatsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {
            "tfidf": 0,
            "hybrid": 0,
            "rag": 0,
            "fallback_hybrid": 0,
            "abstain": 0
        }

        self.latencies = {
            "tfidf": [],
            "hybrid": [],
            "rag": [],
            "fallback_hybrid": [],
            "abstain": []
        }

    def log(self, mode, latency):
        if mode not in self.counts:
            self.counts[mode] = 0
            self.latencies[mode] = []

        self.counts[mode] += 1
        self.latencies[mode].append(latency)

    def summary(self):
        summary = {}

        for mode in self.counts:
            count = self.counts[mode]
            lat_list = self.latencies.get(mode, [])

            avg_latency = sum(lat_list) / len(lat_list) if lat_list else 0

            summary[mode] = {
                "count": count,
                "avg_latency_ms": round(avg_latency * 1000, 2)
            }

        summary["abstention_rate"] = self.abstention_rate()

        return summary

    def abstention_rate(self):
        total = sum(self.counts.values())
        if total == 0:
            return 0
        return round(self.counts["abstain"] / total, 3)