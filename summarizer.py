import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keybert import KeyBERT
from transformers import pipeline

kw_model = KeyBERT()

abstractive_model = pipeline(
    model="facebook/bart-large-cnn",
    device=-1
)

def extractive_summary(text, num_keywords=5):
    """Generate a keyword-based summary using KeyBERT."""
    try:
        keywords = kw_model.extract_keywords(
            text,
            top_n=num_keywords
        )
        return ", ".join([kw for kw, _ in keywords])

    except Exception as e:
        return f"(extractive summary failed: {e})"


def abstractive_summary(
    text,
    model_name="facebook/bart-large-cnn",
    max_tokens=512
):
    if len(text) > 2000:
        text = text[:2000]

    summary = abstractive_model(
        text,
        max_length=200,
        min_length=50,
        do_sample=False
    )

    return summary[0]["summary_text"]
