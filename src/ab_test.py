from src.hybrid_search import HybridSearch
from src.router import QueryRouter
from src.test_query import TEST_QUERIES
import time


def run_ab_test():
    hybrid = HybridSearch(alpha=0.6)
    router = QueryRouter()

    hybrid_latency = []
    routing_latency = []

    hybrid_counts = 0
    routing_counts = {
        "tfidf": 0,
        "hybrid": 0,
        "rag": 0,
        "fallback_hybrid": 0
    }

    print("\n🔹 Running A/B Test...\n")

    for q in TEST_QUERIES:
        query = q["query"]

        # 🔵 System A: Always Hybrid
        start = time.time()
        _ = hybrid.search(query, top_k=5, use_expansion=True)
        hybrid_latency.append(time.time() - start)
        hybrid_counts += 1

        # 🟢 System B: Routing
        start = time.time()
        output = router.route(query)
        routing_latency.append(time.time() - start)

        routing_counts[output["mode"]] += 1

    # 📊 Results
    avg_hybrid_latency = sum(hybrid_latency) / len(hybrid_latency)
    avg_routing_latency = sum(routing_latency) / len(routing_latency)

    print("===== A/B TEST RESULTS =====\n")

    print("🔵 System A (Always Hybrid):")
    print(f"Avg Latency: {round(avg_hybrid_latency * 1000, 2)} ms")

    print("\n🟢 System B (Routing System):")
    print(f"Avg Latency: {round(avg_routing_latency * 1000, 2)} ms")

    print("\nRouting Distribution:")
    for k, v in routing_counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    run_ab_test()