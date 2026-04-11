from src.eval import precision_at_k, mean_reciprocal_rank
from src.tf_idf import TfidfRetriever
from src.hybrid_search import HybridSearch
from src.eval_dataset import load_eval_dataset



def hallucination_rate(outputs):
    total = len(outputs)
    hall = sum(1 for o in outputs if o.get("hallucination", 0) > 0)
    return hall / total if total else 0


def evaluate_system(name, search_fn, dataset):
    p_total = 0
    mrr_total = 0
    abstain_count = 0
    outputs = []

    for q in dataset:
        results = search_fn(q["query"])

        if len(results) == 0:
            abstain_count += 1
            continue

        p = precision_at_k(results, q["relevant_docs"], k=5)
        mrr = mean_reciprocal_rank(results, q["relevant_docs"])

        p_total += p
        mrr_total += mrr

        outputs.append({
            "results": results,
            "relevant_docs": q["relevant_docs"]
        })

    n = len(dataset)

    return {
        "precision@5": round(p_total / n, 3),
        "mrr": round(mrr_total / n, 3),
        "abstention_rate": round(abstain_count / n, 3)
    }


def run_evaluation():
    dataset = load_eval_dataset()

    # TF-IDF
    tfidf = TfidfRetriever()
    tfidf.load()

    def tfidf_search(q):
        return tfidf.search(q, top_k=5)

    # Hybrid
    hybrid = HybridSearch(alpha=0.6)

    def hybrid_search(q):
        results, _ = hybrid.search(q, top_k=5, use_expansion=False)
        return results

    def hybrid_expanded_search(q):
        results, _ = hybrid.search(q, top_k=5, use_expansion=True)
        return results

    print("\n🔹 Evaluating TF-IDF...")
    r1 = evaluate_system("TF-IDF", tfidf_search, dataset)

    print("\n🔹 Evaluating Hybrid...")
    r2 = evaluate_system("Hybrid", hybrid_search, dataset)

    print("\n🔹 Evaluating Hybrid + Query Expansion...")
    r3 = evaluate_system("Hybrid+QE", hybrid_expanded_search, dataset)

    print("\n===== FINAL RESULTS =====")
    print(f"TF-IDF        → {r1}")
    print(f"Hybrid        → {r2}")
    print(f"Hybrid + QE   → {r3}")


if __name__ == "__main__":
    run_evaluation()