def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def chunk_documents(documents, chunk_size=200):
    chunked_docs = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size)

        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "id": f"{doc['id']}_{idx}",
                "title": doc["title"],
                "text": chunk,
                "metadata": doc.get("metadata", {})
            })

    return chunked_docs