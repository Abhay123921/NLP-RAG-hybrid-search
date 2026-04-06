import faiss
import numpy as np
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "processed", "faiss.index")
META_PATH = os.path.join(BASE_DIR, "processed", "faiss_meta.pkl")


class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim) 
        self.documents = None

    def build(self, embeddings, documents):
        print("🔹 Building FAISS index...")
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents = documents

    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "score": float(score),
                "document": self.documents[idx]
            })

        return results