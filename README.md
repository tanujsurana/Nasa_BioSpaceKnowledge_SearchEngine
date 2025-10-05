# 🚀 NASA Bioscience Knowledge Explorer

An AI-powered search engine that explores NASA's bioscience publications and provides intelligent extractive + abstractive summaries.

### 🔍 Features
- AI-powered semantic search (FAISS + Sentence Transformers)
- Extractive and abstractive summarization using Transformers
- Keyword extraction with KeyBERT
- Real-time query interface via Streamlit

### 🧠 Tech Stack
- Python, Streamlit, FAISS, Sentence Transformers, HuggingFace Transformers
- KeyBERT for keyword extraction
- NLTK for preprocessing

### ⚙️ Run Locally
```bash
git clone https://github.com/tanujsurana/Nasa_BioSpaceKnowledge_SearchEngine.git
cd Nasa_BioSpaceKnowledge_SearchEngine
pip install -r requirements.txt
streamlit run app.py
