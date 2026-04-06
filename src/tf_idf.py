from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50000)
        self.doc_vectors = None
        self.documents = None

    def fit(self, documents):
        self.documents = documents
        corpus = [doc["text"] for doc in documents]

        print("🔹 Training TF-IDF...")
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def save(self, path="processed/tfidf.pkl"):
        joblib.dump((self.vectorizer, self.doc_vectors, self.documents), path)

    def load(self, path="processed/tfidf.pkl"):
        self.vectorizer, self.doc_vectors, self.documents = joblib.load(path)

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])

        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "document": self.documents[idx]
            })

        return results