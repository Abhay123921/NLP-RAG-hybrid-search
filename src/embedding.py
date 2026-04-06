from sentence_transformers import SentenceTransformer
from src.utils import normalize

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("🔹 Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents, batch_size=32):
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        embeddings = normalize(embeddings)
        return embeddings

    def encode_query(self, query):
        return self.model.encode([query])