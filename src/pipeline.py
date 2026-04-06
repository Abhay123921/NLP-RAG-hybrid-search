import json
from src.config import OUTPUT_PATH, CHUNK_SIZE, USE_CHUNKING, REMOVE_DUPLICATES

from src.loader import load_data
from src.preprocess import preprocess_documents
from src.chunking import chunk_documents


def remove_duplicates(documents):
    seen = set()
    unique_docs = []

    for doc in documents:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            unique_docs.append(doc)

    return unique_docs


def compute_stats(documents):
    total_docs = len(documents)
    avg_length = sum(len(d["text"].split()) for d in documents) / total_docs

    print(f"Total documents: {total_docs}")
    print(f"Average length: {avg_length:.2f} words")


def save_documents(documents):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(documents, f, indent=2)


def run_pipeline():
    print("🔹 Loading data...")
    documents = load_data()

    print("🔹 Preprocessing...")
    documents = preprocess_documents(documents, mode="transformer")

    if REMOVE_DUPLICATES:
        print("🔹 Removing duplicates...")
        documents = remove_duplicates(documents)

    if USE_CHUNKING:
        print("🔹 Chunking documents...")
        documents = chunk_documents(documents, CHUNK_SIZE)

    print("🔹 Computing statistics...")
    compute_stats(documents)

    print("🔹 Saving processed data...")
    save_documents(documents)

    print("✅ Pipeline completed!")


if __name__ == "__main__":
    run_pipeline()