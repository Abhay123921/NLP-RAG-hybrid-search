def expand_query(query):
    query = query.lower()

    expansions = [query]

    # domain-specific expansions
    if "machine learning" in query:
        expansions += ["ml algorithms", "supervised learning", "unsupervised learning"]

    if "neural" in query or "deep learning" in query:
        expansions += ["neural networks", "deep neural models", "cnn rnn"]

    if "car" in query or "vehicle" in query:
        expansions += ["automobile", "transport vehicles", "cars industry"]

    if "cyber" in query:
        expansions += ["cyber attacks", "network security", "malware attacks"]

    if "nlp" in query or "language" in query:
        expansions += ["natural language processing", "text processing", "language models"]

    return list(set(expansions))