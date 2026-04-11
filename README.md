# 🧠 Intelligent RAG System with Cost-Aware Query Routing

An end-to-end **production-style Retrieval-Augmented Generation (RAG) system** with intelligent query routing, hybrid retrieval, and reliability-aware decision making.

---

## 🚀 Key Features

* 🔀 **Intent-Aware Query Routing**

  * Classifies queries into *simple*, *semantic*, and *complex*
  * Dynamically routes to TF-IDF, Hybrid Search, or RAG

* 🔎 **Hybrid Retrieval (TF-IDF + FAISS)**

  * Combines lexical and semantic search
  * Uses score normalization + weighted fusion

* 🧠 **Query Expansion**

  * Expands queries using domain-specific heuristics
  * Improves recall for semantic queries

* ⚡ **Cost-Aware Execution**

  * Avoids expensive RAG calls for simple queries
  * Reduces latency via adaptive routing

* 🤖 **RAG Pipeline**

  * Retrieves relevant chunks
  * Generates extractive answers
  * Avoids hallucination using quality checks

* 🛡️ **Reliability Layer (CORE CONTRIBUTION)**

  * Faithfulness scoring
  * Hallucination detection
  * Trust score-based decision making
  * **Abstains when confidence is low**

* 🔁 **Feedback Loop**

  * Logs failed queries
  * Enables system improvement over time

---

## 🏗️ System Architecture

```
Query
  ↓
Intent Classifier
  ↓
Router
 ├── TF-IDF (fast path)
 ├── Hybrid Search (semantic)
 └── RAG (complex queries)
        ↓
   Answer Generation
        ↓
   Quality + Faithfulness Check
        ↓
   Vague Query Filter
        ↓
   Trust Score Decision
        ↓
   Answer / Abstain
```

---

## 📊 Evaluation Results

| Model       | Precision@5 | MRR   |
| ----------- | ----------- | ----- |
| TF-IDF      | 0.157       | 0.279 |
| Hybrid      | 0.557       | 0.637 |
| Hybrid + QE | 0.529       | 0.643 |

### 🔥 Key Insights

* ~3.5× improvement in Precision over TF-IDF
* ~2.3× improvement in ranking quality (MRR)
* Query Expansion improves ranking but may introduce noise

---

## 🧪 Reliability Testing

The system was tested on noisy and invalid queries:

| Query Type    | Behavior   |
| ------------- | ---------- |
| Random input  | ✅ Abstains |
| Vague queries | ✅ Abstains |
| Valid queries | ✅ Answers  |

👉 The system **knows when it does not know**, reducing hallucinations.

---

## ⚙️ Tech Stack

* Python
* FAISS (vector search)
* Sentence Transformers (embeddings)
* Scikit-learn (TF-IDF)
* FastAPI (API deployment)
* Streamlit (UI)
* OpenAI API (optional LLM integration)

---

## 📁 Project Structure

```
src/
 ├── router.py
 ├── rag.py
 ├── hybrid_search.py
 ├── tf_idf.py
 ├── embedding.py
 ├── query_expansion.py
 ├── intent_classifier.py
 ├── quality_check.py
 ├── faithfulness.py
 ├── eval.py
 ├── eval_runner.py
 └── ...
```

---

## ▶️ How to Run

### 1. Build Indexes

```bash
python -m src.search
python -m src.semantic_search
```

### 2. Run API

```bash
uvicorn src.api:app --reload
```

### 3. Run Streamlit UI

```bash
streamlit run app.py
```

---

## 🧪 Run Evaluation

```bash
python -m src.eval_runner
```

---

## 🔍 Example Query

```bash
GET /smart_search?query=what is machine learning
```

---

## 💡 Key Learnings

* Hybrid retrieval significantly outperforms lexical search
* Query expansion improves recall but must be controlled
* Faithfulness alone is insufficient — semantic correctness matters
* **Confidence ≠ correctness**
* Reliability layers are critical for production systems

---

## 🎯 Future Improvements

* LLM-based answer generation
* Better intent classification (ML-based)
* Learning-based query expansion
* Reinforcement learning from feedback logs

---

## 👨‍💻 Author

**Abhay Raj Singh**
M.Tech (Machine Learning & Data Science), ISI Kolkata

---



## 🏗️ System Architecture

```mermaid
flowchart TD
    A[User Query] --> B[Intent Classifier]

    B -->|Simple| C[TF-IDF Retrieval]
    B -->|Semantic| D[Hybrid Search (TF-IDF + FAISS)]
    B -->|Complex| E[RAG Pipeline]

    D --> F[Top-K Documents]
    C --> F
    E --> F

    F --> G[Answer Generation]

    G --> H[Quality Score]
    G --> I[Faithfulness Score]

    H --> J[Trust Score]
    I --> J

    J --> K{Decision}

    K -->|High Trust| L[Return Answer]
    K -->|Low Trust| M[Abstain]

    M --> N[Log Failure (Feedback Loop)]
