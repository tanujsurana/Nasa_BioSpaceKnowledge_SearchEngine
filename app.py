import os

# Force CPU / reduce threading issues on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from summarizer import extractive_summary, abstractive_summary


# --------------------------------------------------
# File paths
# --------------------------------------------------

INDEX_FILE = "faiss_index.bin"
EMBED_FILE = "embeddings.npy"
PUBS_FILE = "publications_with_text.csv"


# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------

st.set_page_config(
    page_title="NASA Bioscience Knowledge Explorer",
    layout="wide"
)

st.title("🚀 NASA Bioscience Knowledge Explorer")
st.caption(
    "Explore NASA bioscience publications with AI-powered search and summaries."
)


# --------------------------------------------------
# Check required files
# --------------------------------------------------

required_files = [
    INDEX_FILE,
    EMBED_FILE,
    PUBS_FILE
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.error(
        "⚠️ Required search files are missing: "
        + ", ".join(missing_files)
    )

    st.info(
        "Run `python3 ingest.py` and `python3 vector_store.py` "
        "before starting the application."
    )

    st.stop()


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )


# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(INDEX_FILE)


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

@st.cache_data
def load_embeddings():
    return np.load(EMBED_FILE)


# --------------------------------------------------
# Load publications
# --------------------------------------------------

@st.cache_data
def load_publications():
    return pd.read_csv(PUBS_FILE)


# --------------------------------------------------
# Load resources safely
# --------------------------------------------------

try:
    model = load_embedding_model()
    index = load_faiss_index()
    embeddings = load_embeddings()
    pubs = load_publications()

except Exception as e:
    st.error("❌ Failed to load application resources.")

    # Useful while debugging deployment
    st.exception(e)

    st.stop()


# --------------------------------------------------
# Search interface
# --------------------------------------------------

query = st.text_input(
    "🔎 Enter a search query",
    placeholder="e.g. bone density loss in astronauts"
)


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

if query.strip():

    try:
        # Generate semantic embedding for user query
        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        query_vec = np.asarray(
            query_vec,
            dtype="float32"
        )

        # Retrieve top 3 similar publications
        distances, indices = index.search(
            query_vec,
            3
        )

    except Exception as e:
        st.error("❌ An error occurred while performing semantic search.")
        st.exception(e)
        st.stop()


    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    valid_results = 0

    for idx in indices[0]:

        if idx < 0 or idx >= len(pubs):
            continue

        valid_results += 1

        row = pubs.iloc[idx]

        title = row.get("Title", "Untitled Publication")
        text = row.get("Text", "")

        if pd.isna(text):
            text = ""

        text = str(text)

        with st.expander(f"📄 {title}"):

            # Show article preview
            if text:
                preview_length = min(1000, len(text))

                st.write(
                    text[:preview_length]
                    + ("..." if len(text) > preview_length else "")
                )

            else:
                st.warning("No publication text is available.")


            # --------------------------------------------------
            # Generate summaries
            # --------------------------------------------------

            if text.strip():

                try:
                    summary_ext = extractive_summary(text)

                    st.markdown(
                        "**Extractive Summary (keywords):**"
                    )

                    st.write(summary_ext)

                except Exception as e:
                    st.warning(
                        "Could not generate the extractive summary."
                    )


                try:
                    summary_abs = abstractive_summary(text)

                    st.markdown(
                        "**Abstractive Summary:**"
                    )

                    st.write(summary_abs)

                except Exception as e:
                    st.warning(
                        "Could not generate the abstractive summary."
                    )


            # --------------------------------------------------
            # Publication link
            # --------------------------------------------------

            if "Link" in pubs.columns:

                link = row.get("Link")

                if pd.notna(link) and str(link).strip():
                    st.markdown(
                        f"[🔗 Read full article]({link})"
                    )


    if valid_results == 0:
        st.warning(
            "No matching publications were found."
        )
