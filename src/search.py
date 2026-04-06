import json
from src.tf_idf import TfidfRetriever

PROCESSED_PATH = "processed/documents.json"


def load_documents():
    with open(PROCESSED_PATH, "r") as f:
        return json.load(f)


def build_index():
    docs = load_documents()

    retriever = TfidfRetriever()
    retriever.fit(docs)
    retriever.save()

    print("✅ TF-IDF index built and saved!")


def run_search():
    retriever = TfidfRetriever()
    retriever.load()

    while True:
        query = input("\n🔍 Enter query (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = retriever.search(query, top_k=5)

        print("\nTop Results:\n")
        for i, res in enumerate(results):
            print(f"{i+1}. ID: {res['document']['id']}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Text: {res['document']['text'][:150]}...\n")


if __name__ == "__main__":
    # build_index()
    run_search()