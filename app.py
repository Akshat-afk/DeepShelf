import os
import streamlit as st
import faiss
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from langdetect import detect

HF_REPO = "AKKI-AFK/deepshelf-data"

# Load data
books_file = hf_hub_download(repo_id=HF_REPO, filename="booksummaries.txt", repo_type="dataset")
faiss_file = hf_hub_download(repo_id=HF_REPO, filename="faiss_index.bin", repo_type="dataset")

df = pd.read_csv(books_file, delimiter="\t")
index = faiss.read_index(faiss_file)

# Load models
encoder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

MAX_QUERY_LENGTH = 300  # Set max input length

def recommend_books(query):
    """Recommends books based on query"""
    
    # 1. Check for empty input
    query = query.strip()
    if not query:
        st.warning("⚠️ Please enter a valid query.")
        return []

    # 2. Check for excessive input length
    if len(query) > MAX_QUERY_LENGTH:
        st.warning(f"⚠️ Your query is too long! Please limit it to {MAX_QUERY_LENGTH} characters.")
        return []

    # 3. Language detection
    try:
        lang = detect(query)
        if lang != "en":
            st.warning("🌍 Currently, only English queries are supported.")
            return []
    except:
        st.error("⚠️ Invalid input. Please enter a meaningful query.")
        return []

    # 4. Token length check (512 tokens max)
    query_tokens = encoder.tokenizer.tokenize(query)[:512]
    query = encoder.tokenizer.convert_tokens_to_string(query_tokens)

    # Encode query and search FAISS index
    search_vector = encoder.encode(query)
    search_vector = np.array([search_vector])
    faiss.normalize_L2(search_vector)

    distances, ann = index.search(search_vector, k=50)
    
    if len(ann) == 0 or np.all(ann == -1):
        st.warning("🔍 No relevant books found. Try refining your query.")
        return []

    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    merge = pd.merge(results, df, left_on='ann', right_index=True)
    merge['Query'] = query

    # Cross-encoder ranking
    pairs = list(zip(merge['Query'], merge['summary']))
    scores = cross_encoder.predict(pairs)
    merge['score'] = scores

    df_sorted = merge.iloc[merge["score"].argsort()][::-1]
    
    return df_sorted[["title", "summary"]][:5].to_dict(orient="records")

# Streamlit UI
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

query = st.text_area("🔍 Enter a book description (e.g., 'A dark fantasy with drama')", 
                     help="Use keywords to describe your ideal book!", 
                     max_chars=MAX_QUERY_LENGTH)  # Enforces max input length in UI

if st.button("✨ Recommend Books", help="Click to get personalized book recommendations!"):
    recommendations = recommend_books(query)
    
    if recommendations:
        st.markdown("## 📚 Recommended Books:")
        for rec in recommendations:
            st.markdown(f"""
                <div class="book-container">
                    <div class="book-title">📖 {rec["title"]}</div>
                    <div class="book-summary">{rec["summary"]}</div>
                </div>
            """, unsafe_allow_html=True)
