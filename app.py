import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
import os
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from summarizer import extractive_summary, abstractive_summary

# File paths
INDEX_FILE = "faiss_index.bin"
EMBED_FILE = "embeddings.npy"
PUBS_FILE = "publications_with_text.csv"

st.set_page_config(page_title="NASA Bioscience Knowledge Explorer", layout="wide")
st.title("🚀 NASA Bioscience Knowledge Explorer")
st.caption("Explore NASA bioscience publications with AI-powered search and summaries.")

# Ensure index and data exist
if not (os.path.exists(INDEX_FILE) and os.path.exists(EMBED_FILE) and os.path.exists(PUBS_FILE)):
    st.error("⚠️ No FAISS index found. Please run `python3 ingest.py` and `python3 vector_store.py` first.")
    st.stop()

# Load FAISS + embeddings + pubs
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(EMBED_FILE)
pubs = pd.read_csv(PUBS_FILE)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Search box
query = st.text_input("🔎 Enter a search query (e.g., 'bone density loss in astronauts')")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), 3)

    for idx in I[0]:
        if idx >= len(pubs): 
            continue
        row = pubs.iloc[idx]

        with st.expander(f"📄 {row['Title']}"):
            st.write(row["Text"][:1000] + "...")
            summary_ext = extractive_summary(row["Text"])
            summary_abs = abstractive_summary(row["Text"])
            st.markdown("**Extractive Summary (keywords):** " + summary_ext)
            st.markdown("**Abstractive Summary:** " + summary_abs)
            if "Link" in row:
                st.markdown(f"[🔗 Read full article]({row['Link']})")

