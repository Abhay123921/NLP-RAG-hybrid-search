import time
from src.rag import RAGPipeline

rag_pipeline = RAGPipeline()
@app.get("/search")
def search(query: str, k: int = 5):
    start = time.time()

    results = search_engine.search(query, top_k=k)

    latency = time.time() - start

    return {
        "query": query,
        "latency_ms": round(latency * 1000, 2),
        "results": [
            {
                "score": r["score"],
                "text": r["document"]["text"]
            }
            for r in results
        ]
    }

@app.get("/rag")
def rag_search(query: str):
    start = time.time()

    output = rag_pipeline.generate_answer(query)

    latency = time.time() - start

    return {
        "query": query,
        "latency_ms": round(latency * 1000, 2),
        "answer": output["answer"],
        "results": [
            {
                "score": r["score"],
                "text": r["document"]["text"]
            }
            for r in output["documents"]
        ]
    }