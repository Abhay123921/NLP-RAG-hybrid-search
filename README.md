# 🔍 Hybrid Search + RAG Pipeline

An end-to-end **hybrid information retrieval system** combining lexical and semantic search, enhanced with query expansion and Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🔎 TF-IDF based lexical search
- 🧠 Semantic search using Sentence Transformers + FAISS
- 🔗 Hybrid ranking (weighted fusion)
- 🔄 Query expansion (analyzed & optimized)
- 🤖 RAG pipeline for context-aware answer generation
- 📊 Evaluation using Precision@K and Mean Reciprocal Rank (MRR)
- 🌐 FastAPI-based API

---

## 🧠 System Architecture


User Query
↓
Query Expansion
↓
Hybrid Retrieval (TF-IDF + FAISS)
↓
Top-K Documents
↓
RAG (LLM + Context)
↓
Final Answer


---

## 📊 Results

| Method        | Precision@5 | MRR  |
|--------------|------------|------|
| TF-IDF       | 0.533      | 0.694 |
| Hybrid       | 0.550      | 0.688 |
| Hybrid + QE  | 0.483      | 0.625 |

> Observation: Naive query expansion can introduce noise and degrade retrieval performance.

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/NLP-RAG-hybrid-search.git
cd NLP-RAG-hybrid-search

pip install -r requirements.txt
▶️ Run Pipeline
python -m src.pipeline
python -m src.search
python -m src.sementic_search
📊 Evaluate
python -m src.eval_runner
🌐 Run API
uvicorn src.api:app --reload


📌 API Endpoints
/search?query=... → returns top documents
/rag?query=... → returns generated answer + documents


🧠 Key Learnings
Hybrid retrieval improves over lexical baselines
Query expansion requires careful design
RAG enables context-aware answer generation


🚀 Future Improvements
LLM-based query expansion
Reranking models
UI (Streamlit)



👨‍💻 Author
Abhay Raj Singh