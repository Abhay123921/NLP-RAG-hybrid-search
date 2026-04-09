# 🧠 Intelligent Search System with Cost-Aware Query Routing

## 🚀 Overview

This project implements a **production-style Retrieval-Augmented Generation (RAG) system** with **intent-aware query routing** to optimize search relevance, latency, and computational cost.

Instead of applying the same retrieval pipeline to all queries, the system dynamically selects the most efficient strategy based on query complexity.

---

## 🎯 Key Features

* 🔍 **Hybrid Retrieval System**

  * TF-IDF (lexical search)
  * FAISS with dense embeddings (semantic search)

* 🤖 **RAG Pipeline**

  * Retrieves top-K documents
  * Generates context-aware responses (LLM-ready architecture)

* 🧠 **Intent-Aware Query Routing**

  * Classifies queries into:

    * Simple → TF-IDF
    * Semantic → Hybrid Search
    * Complex → RAG

* ⚠️ **Confidence-Based Fallback**

  * Low-confidence queries routed to hybrid search for robustness

* 📊 **Evaluation & Metrics**

  * Precision@K, Mean Reciprocal Rank (MRR)
  * A/B testing (baseline vs routing system)
  * Latency tracking per query type

* ⚡ **Cost Optimization**

  * Reduces unnecessary use of expensive pipelines (RAG/LLM)

* 🌐 **API + UI**

  * FastAPI backend
  * Streamlit interactive interface

---

## 🏗️ System Architecture

```
User Query
   ↓
Intent Classifier
   ↓
Confidence Check
   ↓
Routing Layer
   ├── TF-IDF (Simple Queries)
   ├── Hybrid Search (Semantic Queries)
   └── RAG Pipeline (Complex Queries)
   ↓
Final Results / Answer
```

---

## 📂 Project Structure

```
NLP/
├── data/
├── processed/
├── src/
│   ├── tf_idf.py
│   ├── embedding.py
│   ├── faiss_index.py
│   ├── hybrid_search.py
│   ├── query_expansion.py
│   ├── rag.py
│   ├── intent_classifier.py
│   ├── router.py
│   ├── stats.py
│   ├── eval.py
│   ├── eval_runner.py
│   ├── ab_test.py
│   └── api.py
├── app.py   # Streamlit UI
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
git clone <your-repo-url>
cd NLP

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Run the System

### 1️⃣ Data Pipeline

```bash
python -m src.pipeline
```

### 2️⃣ Build Indexes

```bash
python -m src.search          # TF-IDF
python -m src.sementic_search # FAISS
```

### 3️⃣ Start API

```bash
uvicorn src.api:app --reload
```

### 4️⃣ Launch UI (Optional)

```bash
streamlit run app.py
```

---

## 🔍 API Endpoints

* `/search` → Basic retrieval
* `/rag` → RAG-based answer
* `/smart_search` → Intent-aware routing
* `/stats` → Routing distribution + latency

---

## 📊 A/B Testing Results

| System         | Avg Latency    |
| -------------- | -------------- |
| Always Hybrid  | 52.34 ms       |
| Routing System | **36.49 ms** ✅ |

### 🔥 Key Insight:

* ~30% reduction in latency using routing
* Majority of queries handled by lightweight retrieval
* Expensive RAG used only when necessary

---

## 📈 Routing Distribution Example

```
TF-IDF: 5 queries
Hybrid: 3 queries
RAG: 1 query
Fallback: 3 queries
```

---

## 🧠 Key Learnings

* Naive query expansion can degrade performance due to noise
* Hybrid retrieval improves relevance over lexical methods
* Intent-based routing significantly reduces latency and cost
* Most queries do not require full RAG pipeline

---

## 🎯 Future Improvements

* Integrate real LLM (OpenAI / Ollama)
* Train ML-based intent classifier
* Add re-ranking (cross-encoder)
* Deploy using Docker / AWS

---

## 🏆 Resume Highlights

* Built a hybrid retrieval system (TF-IDF + FAISS)
* Designed intent-aware routing for cost and latency optimization
* Achieved ~30% latency reduction via A/B testing
* Implemented full RAG pipeline with evaluation metrics (MRR, Precision@K)

---

## 👨‍💻 Author

**Abhay Raj Singh**
ISI Kolkata M.Tech CS Student(Machine Learning & Data Science)

---
