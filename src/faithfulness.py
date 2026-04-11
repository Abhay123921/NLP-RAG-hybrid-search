from src.embedding import EmbeddingModel
import numpy as np

embedder = EmbeddingModel()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def faithfulness_score(answer, chunks):
    answer_emb = embedder.encode_query(answer)[0]

    scores = []
    for chunk in chunks:
        chunk_emb = embedder.encode_query(chunk)[0]
        scores.append(cosine(answer_emb, chunk_emb))

    return np.mean(scores) if scores else 0


# 🔥 NEW: hallucination metric
def hallucination_score(answer, chunks):
    """
    1 → hallucinated
    0 → grounded
    """
    faith = faithfulness_score(answer, chunks)

    # strong mismatch → hallucination
    if faith < 0.4:
        return 1

    # medium mismatch → partial hallucination
    if faith < 0.6:
        return 0.5

    return 0