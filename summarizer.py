# summarizer.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from keybert import KeyBERT
from transformers import pipeline


# --------------------------------------------------
# Load KeyBERT lazily
# --------------------------------------------------

@st.cache_resource
def load_keybert_model():
    return KeyBERT(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# Load summarization model lazily
# --------------------------------------------------

@st.cache_resource
def load_summarization_pipeline():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=-1
    )


# --------------------------------------------------
# Extractive summary / keywords
# --------------------------------------------------

def extractive_summary(text):

    if not text or not text.strip():
        return "No text available."

    kw_model = load_keybert_model()

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )

    if not keywords:
        return "No keywords found."

    return ", ".join(
        keyword
        for keyword, score in keywords
    )


# --------------------------------------------------
# Abstractive summary
# --------------------------------------------------

def abstractive_summary(text):

    if not text or not text.strip():
        return "No text available."

    summarizer = load_summarization_pipeline()

    # Keep input reasonably small for BART
    text = text[:4000]

    result = summarizer(
        text,
        max_length=150,
        min_length=40,
        do_sample=False
    )

    if not result:
        return "Could not generate summary."

    return result[0]["summary_text"]
