import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

data_path = "data/processed_products.csv"
index_path = "models/faiss_index.bin"
metadeta_path = "models/product_metadata.pkl"

#loading processed data
df = pd.read_csv(data_path)

texts = df["combined_text"].astype(str).tolist()

#loading embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#creating embedding 
embeddings = model.encode(
    texts,
    show_progress_bar = True,
    batch_size = 64,
    convert_to_numpy = True
)

#normalizing embeddings
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

#building FAISS index
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

#saving FAISS index
faiss.write_index(index, index_path)

metadata = df[[
    "id",
    "title",
    "category_name",
    "price",
    "stars",
    "reviews",
    "isBestSeller",
    "imgUrl",
    "productURL"
]]

with open(metadeta_path, "wb") as f:
    pickle.dump(metadata, f)

print("Faiss index built")