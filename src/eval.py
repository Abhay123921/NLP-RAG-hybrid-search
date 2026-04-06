def precision_at_k(results, relevant_docs, k=5):
    retrieved = results[:k]
    relevant = 0

    for r in retrieved:
        if r["document"]["id"] in relevant_docs:
            relevant += 1

    return relevant / k


def mean_reciprocal_rank(results, relevant_docs):
    for i, r in enumerate(results):
        if r["document"]["id"] in relevant_docs:
            return 1 / (i + 1)
    return 0