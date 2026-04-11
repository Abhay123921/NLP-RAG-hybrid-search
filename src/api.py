import time
from fastapi import FastAPI

from src.rag import RAGPipeline
from src.router import QueryRouter
from src.hybrid_search import HybridSearch  

app = FastAPI()  

# Initialize systems
rag_pipeline = RAGPipeline()
router = QueryRouter()
search_engine = HybridSearch(alpha=0.6)  


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


# 🔥 ADD THIS NEW ENDPOINT
@app.get("/smart_search")
def smart_search(query: str):
    start = time.time()

    output = router.route(query)

    latency = time.time() - start

    response = {
        "query": query,
        "intent": output["intent"],
        "confidence": output["confidence"],
        "mode": output["mode"],
        "latency_ms": round(latency * 1000, 2),
    }

    if output["mode"] == "rag":
        response["answer"] = output["answer"]

    response["results"] = [
        {
            "score": r["score"],
            "text": r["document"]["text"]
        }
        for r in output["results"]
    ]

    return response

@app.get("/stats")
def get_stats():
    return router.stats.summary()