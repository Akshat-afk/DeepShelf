import os
import streamlit as st
import faiss
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from streamlit_lottie import st_lottie

from huggingface_hub import hf_hub_download

HF_REPO = "AKKI-AFK/deepshelf-data"

books_file = hf_hub_download(repo_id=HF_REPO, filename="booksummaries.txt", repo_type="dataset")
faiss_file = hf_hub_download(repo_id=HF_REPO, filename="faiss_index.bin", repo_type="dataset")

df = pd.read_csv(books_file, delimiter="\t")
index = faiss.read_index(faiss_file)

encoder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

loading_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_pNx6yH.json")
book_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_4yofo7QwIT.json")

def recommend_books(query):
    search_vector = encoder.encode(query)
    search_vector = np.array([search_vector])
    faiss.normalize_L2(search_vector)

    distances, ann = index.search(search_vector, k=50)
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    merge = pd.merge(results, df, left_on='ann', right_index=True)
    merge['Query'] = query

    pairs = list(zip(merge['Query'], merge['summary']))
    scores = cross_encoder.predict(pairs)
    merge['score'] = scores

    df_sorted = merge.iloc[merge["score"].argsort()][::-1]
    return df_sorted[["title", "summary"]][:5].to_dict(orient="records")

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

st.markdown('<div class="title">📖 AI-Powered Novel Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Find the best books based on your preferences!</div>', unsafe_allow_html=True)

st_lottie(book_animation, height=150, key="book_anim")

query = st.text_input("🔍 Enter a book description (e.g., 'A dark fantasy with drama')", help="Use keywords to describe your ideal book!")

if st.button("✨ Recommend Books", help="Click to get personalized book recommendations!"):
    if query:
        st_lottie(loading_animation, height=120, key="loading_anim")
        recommendations = recommend_books(query)
        
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

