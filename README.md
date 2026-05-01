# Semantic Product Search Engine

A machine learning-powered product search system that understands **user intent** instead of relying on exact keyword matches. Built using **Sentence-BERT embeddings + FAISS vector search**, enabling fast and relevant retrieval over large product datasets.

---

## Demo

Search queries like:

- "cheap smartphone with good battery"  
- "gaming laptop"  
- "wireless earbuds noise cancellation"  

Returns semantically relevant products even when exact keywords don’t match.

---

## Problem Statement

Traditional keyword-based search (TF-IDF, BM25) fails to capture semantic meaning.

Example:  
Query: *"budget phone"*  
Keyword search fails if product titles don’t contain the word "budget".

This project solves this using vector embeddings that capture meaning beyond keywords.

---

## System Architecture


User Query
↓
Sentence-BERT Embedding
↓
FAISS Vector Search
↓
Top-K Similar Products
↓
Streamlit UI


---

## Tech Stack

- NLP Model: Sentence-Transformers (`all-MiniLM-L6-v2`)
- Vector Database: FAISS
- Frontend: Streamlit
- Data Processing: Pandas, NumPy
- Baseline Model: TF-IDF (for evaluation)

---
```
## Project Structure


semantic-product-search/
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ ├── amazon_products.csv
│ ├── amazon_categories.csv
│ └── processed_products.csv
│
├── src/
│ ├── preprocess.py
│ ├── build_index.py
│ ├── search.py
│ └── evaluate.py
│
├── models/
│ ├── faiss_index.bin
│ └── product_metadata.pkl
│
├── requirements.txt
└── README.md
```
---

## How It Works

1. Preprocess data (merge product + category, create combined text)
2. Generate embeddings using Sentence-BERT
3. Store embeddings in FAISS index
4. Convert user query to embedding
5. Retrieve top-K similar products using cosine similarity

---

## Evaluation

Compared semantic search against a TF-IDF baseline using a proxy relevance metric.

- Average TF-IDF Score: 0.52  
- Average Embedding Score: 0.64  
- Improvement: ~23%  

This shows semantic search improves relevance over keyword-based methods.

---

## Running the Project

### 1. Install dependencies


pip install -r requirements.txt


### 2. Preprocess data


python src/preprocess.py


### 3. Build FAISS index


python src/build_index.py


### 4. Run search (CLI)


python src/search.py


### 5. Launch UI


streamlit run app/streamlit_app.py


---

## Key Features

- Semantic search (understands intent, not just keywords)
- Fast retrieval using FAISS
- End-to-end ML pipeline
- Interactive UI with real-time results
- Quantitative evaluation vs baseline

---

## Learnings

- Semantic vs keyword search
- Vector embeddings
- FAISS similarity search
- End-to-end ML pipeline design
- Practical evaluation techniques

---

## Future Improvements

- Use advanced FAISS indexes (IVF, HNSW)
- Add caching for faster queries
- Hybrid ranking (semantic + popularity)
- Deploy as API (FastAPI)

---

## Author

Atharv Nautiyal

---
