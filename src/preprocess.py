import pandas as pd

products = pd.read_csv("data/amazon_products.csv")
categories = pd.read_csv("data/amazon_categories.csv")

# print(products.isnull().sum())
# print(categories.isnull().sum())

products.dropna(inplace = True)

# print(products.columns)
# print(categories.columns)

df = products.merge(categories, left_on="category_id", right_on = "id", how="left")

df = df[[
    "title",
    "category_name",
    "price",
    "stars",
    "reviews",
    "isBestSeller",
    "imgUrl",
    "productURL"
]]

df.dropna(subset=["title"], inplace=True)

df["combined_text"] = (
    df["title"].astype(str) + " " +
    df["category_name"].astype(str)
)

df.loc[df["isBestSeller"] == True, "combined_text"] += "bestseller"

df = df.sample(50000, random_state=42)

df.reset_index(drop=True, inplace=True)
df["id"] = df.index

df.to_csv("data/processed_products.csv", index=False)

print("Preprocessing over")