import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.search import search

st.set_page_config(page_title="Semantic Product Search", layout="wide")

st.title("🔍 Semantic Product Search")
st.markdown("Search products using natural language (not just keywords)")

query = st.text_input("Enter your search query:")

if query:
    results = search(query, top_k=10)

    st.subheader("Results")

    cols = st.columns(2)

    for i, r in enumerate(results):
        with cols[i % 2]:
            st.markdown(f"### {r['title']}")
            st.image(r["image"], width=150)
            st.write(f" Rating: {r['rating']}")
            st.write(f" Price: {r['price']}")
            st.write(f" Category: {r['category']}")
            st.markdown(f"[View Product]({r['url']})")
            st.markdown("---")