import json
from src.embedding import EmbeddingModel
from src.faiss_index import FaissIndex
from src.utils import normalize

PROCESSED_PATH = "processed/documents.json"


def load_documents():
    with open(PROCESSED_PATH, "r") as f:
        return json.load(f)


def build_index():
    docs = load_documents()

    embedder = EmbeddingModel()
    embeddings = embedder.encode_documents(docs)

    dim = embeddings.shape[1]

    index = FaissIndex(dim)
    index.build(embeddings, docs)
    index.save()

    print("✅ FAISS index built and saved!")


def run_search():
    embedder = EmbeddingModel()

    index = FaissIndex(dim=384)  # MiniLM dimension
    index.load()

    while True:
        query = input("\n🔍 Enter query (or 'exit'): ")
        if query.lower() == "exit":
            break

        query_embedding = embedder.encode_query(query)
        query_embedding = normalize(query_embedding)
        results = index.search(query_embedding, top_k=5)

        print("\nTop Results:\n")
        for i, res in enumerate(results):
            print(f"{i+1}. ID: {res['document']['id']}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Text: {res['document']['text'][:150]}...\n")             


if __name__ == "__main__":
    build_index()
    run_search()