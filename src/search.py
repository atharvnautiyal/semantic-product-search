import faiss
import pickle 
import numpy as np
from sentence_transformers import SentenceTransformer

index_path = "models/faiss_index.bin"
metadata_path = "models/product_metadata.pkl"

#loading FAISS index
index = faiss.read_index(index_path)

#loading metadata
with open (metadata_path, "rb") as f:
    metadata = pickle.load(f)

#loading embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, top_k=5):

    #convert query to embedding 
    query_embedding = model.encode([query], convert_to_numpy=True)

    #normalize 
    faiss.normalize_L2(query_embedding)

    #seach
    scores, indices = index.search(query_embedding, top_k)

    results=[]

    for i, idx in enumerate(indices[0]):
        product = metadata.iloc[idx]

        results.append({
            "title": product["title"],
            "category": product["category_name"],
            "price": product["price"],
            "rating": product["stars"],
            "score": float(scores[0][i]),
            "url": product["productURL"],
            "image": product["imgUrl"]
        })

    return results

if __name__ == "__main__":
    query = input("Enter search query: ")

    results = search(query, top_k = 5)

    print("\nTop Results:\n")
    for r in results:
        print(f"{r['title']} | {r['rating']} | {r['price']}")