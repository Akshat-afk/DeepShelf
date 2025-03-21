import os
import streamlit as st
import faiss
import pandas as pd
import numpy as np
import requests
import torch
import asyncio
import re
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from langdetect import detect

# Ensure compatibility with event loops across OS
if os.name == "nt":  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

HF_REPO = "AKKI-AFK/deepshelf-data"

books_file = hf_hub_download(repo_id=HF_REPO, filename="booksummaries.txt", repo_type="dataset")
faiss_file = hf_hub_download(repo_id=HF_REPO, filename="faiss_index.bin", repo_type="dataset")

df = pd.read_csv(books_file, delimiter="\t")
index = faiss.read_index(faiss_file)

encoder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

request_times = []  # Track request timestamps

@st.cache_data
async def recommend_books(query):
    query = sanitize_input(query)
    if len(query) > 200:
        st.warning("⚠️ Query is too long. Please keep it under 200 characters.")
        return []
    
    if len(query) < 3:
        st.warning("⚠️ Query is too short. Please provide more details.")
        return []
    
    try:
        lang = detect(query)
        if lang != "en":
            st.warning("⚠️ Non-English query detected. Results may not be accurate.")
    except:
        st.warning("⚠️ Could not detect language. Ensure proper input.")
    
    search_vector = encoder.encode(query)
    search_vector = np.array([search_vector])
    faiss.normalize_L2(search_vector)

    distances, ann = index.search(search_vector, k=50)
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    merge = pd.merge(results, df, left_on='ann', right_index=True)
    merge['Query'] = query

    pairs = list(zip(merge['Query'], merge['summary']))
    scores = await asyncio.get_event_loop().run_in_executor(None, cross_encoder.predict, pairs)
    merge['score'] = scores

    df_sorted = merge.iloc[merge["score"].argsort()][::-1]
    return df_sorted[["title", "summary"]][:5].to_dict(orient="records")

def sanitize_input(text):
    """Sanitize input by removing special characters and excessive spaces."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def rate_limit():
    """Rate-limiting function to prevent excessive queries."""
    global request_times
    current_time = time.time()
    request_times = [t for t in request_times if current_time - t < 10]  # Keep only recent requests within 10 seconds
    if len(request_times) >= 5:
        st.error("⚠️ Too many requests. Please wait a few seconds before trying again.")
        return False
    request_times.append(current_time)
    return True

st.set_page_config(page_title="DeepShelf", page_icon="📚", layout="wide")

st.markdown("""
    <style>
        body {background-color: #1E1E1E; color: white;}
        .title {text-align: center; font-size: 3em; font-weight: bold; color: #E6A400;}
        .subtext {text-align: center; font-size: 1.2em; color: #AAAAAA;}
        .recommend-btn {text-align: center;}
        .book-container {border-radius: 10px; padding: 20px; margin: 10px; background: #2E2E2E; box-shadow: 2px 2px 10px #00000050;}
        .book-title {font-size: 1.5em; font-weight: bold; color: #FFD700;}
        .book-summary {font-size: 1em; color: #CCCCCC;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📖 DeepShelf</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Find the best books based on your preferences!</div>', unsafe_allow_html=True)

query = st.text_input("🔍 Enter a book description (e.g., 'A dark fantasy with drama')", max_chars=200, help="Use keywords to describe your ideal book!")

if st.button("✨ Recommend Books", help="Click to get personalized book recommendations!"):
    if rate_limit():
        if query:
            with st.spinner("🔍 Searching for the best books..."):
                recommendations = asyncio.run(recommend_books(query))
            
            st.markdown("## 📚 Recommended Books:")
            for rec in recommendations:
                st.markdown(f"""
                    <div class="book-container">
                        <div class="book-title">📖 {rec["title"]}</div>
                        <div class="book-summary">{rec["summary"]}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a query.")
