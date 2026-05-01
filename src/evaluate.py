import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

df = pd.read_csv("data/processed_products.csv")

texts = df["combined_text"].astype(str).tolist()

#======tfidf baseline======
tfidf = TfidfVectorizer(max_features = 10000)
tfidf_matrix = tfidf.fit_transform(texts)

#======embedding model======
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

#======test queries======
queries = [
    "cheap smartphone",
    "gaming laptop",
    "wireless earbuds",
    "kitchen appliances",
    "office chair ergonomic"
]

def tfidf_search(query, top_k=5):
    q_vec = tfidf.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    return scores.argsort()[-top_k:][::-1]

def embedding_search(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return indices[0]

#======simple evaluation======
def relevance_score(indices, keyword):
    score = 0
    for idx in indices:
        title = df.iloc[idx]["title"].lower()
        if keyword in title:
            score += 1
    return score / len(indices)

results = []

for q in queries:
    keyword = q.split()[0]  # simple proxy

    tfidf_idx = tfidf_search(q)
    emb_idx = embedding_search(q)

    tfidf_score = relevance_score(tfidf_idx, keyword)
    emb_score = relevance_score(emb_idx, keyword)

    results.append({
        "query": q,
        "tfidf": tfidf_score,
        "embedding": emb_score
    })

#======final metric======
df_results = pd.DataFrame(results)

avg_tfidf = df_results["tfidf"].mean()
avg_emb = df_results["embedding"].mean()

improvement = ((avg_emb - avg_tfidf) / avg_tfidf) * 100 if avg_tfidf != 0 else 0

print("\nEvaluation Results:")
print(df_results)

print("\nAverage TF-IDF:", round(avg_tfidf, 3))
print("Average Embedding:", round(avg_emb, 3))
print(f"Improvement: {improvement:.2f}%")