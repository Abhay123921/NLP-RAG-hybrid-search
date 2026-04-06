from src.eval import precision_at_k, mean_reciprocal_rank
from src.tf_idf import TfidfRetriever
from src.hybrid_search import HybridSearch
from src.test_query import TEST_QUERIES


def evaluate_system(name, search_fn):
    p_total = 0
    mrr_total = 0

    for q in TEST_QUERIES:
        results = search_fn(q["query"])

        p = precision_at_k(results, q["relevant_docs"], k=5)
        mrr = mean_reciprocal_rank(results, q["relevant_docs"])

        p_total += p
        mrr_total += mrr

    n = len(TEST_QUERIES)

    return round(p_total/n, 3), round(mrr_total/n, 3)


def run_evaluation():
    # TF-IDF
    tfidf = TfidfRetriever()
    tfidf.load()

    def tfidf_search(q):
        return tfidf.search(q, top_k=5)

    # Hybrid (without query expansion)
    hybrid = HybridSearch(alpha=0.6)

    def hybrid_search(q):
        return hybrid.search(q, top_k=5, use_expansion=False)

    # 🔥 Hybrid + Query Expansion (NEW)
    def hybrid_expanded_search(q):
        return hybrid.search(q, top_k=5, use_expansion=True)

    print("\n🔹 Evaluating TF-IDF...")
    p1, mrr1 = evaluate_system("TF-IDF", tfidf_search)

    print("\n🔹 Evaluating Hybrid...")
    p2, mrr2 = evaluate_system("Hybrid", hybrid_search)

    print("\n🔹 Evaluating Hybrid + Query Expansion...")
    p3, mrr3 = evaluate_system("Hybrid+QE", hybrid_expanded_search)

    print("\n===== FINAL RESULTS =====")
    print(f"TF-IDF        → Precision@5: {p1}, MRR: {mrr1}")
    print(f"Hybrid        → Precision@5: {p2}, MRR: {mrr2}")
    print(f"Hybrid + QE   → Precision@5: {p3}, MRR: {mrr3}")


if __name__ == "__main__":
    run_evaluation()