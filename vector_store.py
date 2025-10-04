import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def build_index():
    print("Loading publications_with_text.csv...")
    df = pd.read_csv("publications_with_text.csv")
    
    print(f"Encoding {len(df)} documents...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Text"].tolist(), show_progress_bar=True)
    
    # Save embeddings
    np.save("embeddings.npy", embeddings)
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    
    faiss.write_index(index, "faiss_index.bin")
    print("✅ Index built and saved as faiss_index.bin!")

if __name__ == "__main__":
    build_index()

