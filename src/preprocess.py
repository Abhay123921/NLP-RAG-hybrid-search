import re

def clean_text_tfidf(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()


def clean_text_transformer(text: str) -> str:
    # minimal cleaning
    return text.strip()


def preprocess_documents(documents, mode="transformer"):
    cleaned_docs = []

    for doc in documents:
        text = doc["text"]

        if mode == "tfidf":
            text = clean_text_tfidf(text)
        else:
            text = clean_text_transformer(text)

        cleaned_docs.append({
            **doc,
            "text": text
        })

    return cleaned_docs