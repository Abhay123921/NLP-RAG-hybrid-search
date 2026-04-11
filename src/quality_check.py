from src.faithfulness import faithfulness_score, hallucination_score

def quality_score(answer, chunks, query_type="general"):
    faith = faithfulness_score(answer, chunks)
    halluc = hallucination_score(answer, chunks)

    contradiction = 1 - halluc  # simple proxy
    boundary = 1
    completeness = min(1.0, len(answer.split()) / 50)
    citation = 1 if len(chunks) > 0 else 0
    tone = 1

    weights = get_weights(query_type)

    score = (
        weights["hallucination"] * (1 - halluc) +
        weights["faith"] * faith +
        weights["contradiction"] * contradiction +
        weights["boundary"] * boundary +
        weights["completeness"] * completeness +
        weights["citation"] * citation +
        weights["tone"] * tone
    )

    return score, faith, halluc


def get_weights(query_type):
    if query_type == "billing":
        return {
            "hallucination": 0.5,
            "faith": 0.2,
            "contradiction": 0.1,
            "boundary": 0.1,
            "completeness": 0.05,
            "citation": 0.03,
            "tone": 0.02
        }

    return {
        "hallucination": 0.35,
        "faith": 0.25,
        "contradiction": 0.15,
        "boundary": 0.10,
        "completeness": 0.08,
        "citation": 0.05,
        "tone": 0.02
    }