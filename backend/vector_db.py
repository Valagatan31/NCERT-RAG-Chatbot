from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


# Paths

BASE_PATH = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend\vector_db"
os.makedirs(BASE_PATH, exist_ok=True)

CHUNKS_FILE = "chunks.txt"
MODEL_PATH = r"C:\Users\Admin\Desktop\hugging_face\all-distilroberta-v1"


# Load chunks

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    text = f.read()

raw_chunks = text.split("===== Chunk ")
chunks = [c.strip() for c in raw_chunks if c.strip()]
print(f"Total chunks found: {len(chunks)}")


# Load embedding model

model = SentenceTransformer(MODEL_PATH)


# Generate embeddings

embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]
print(f"Embeddings shape: {embeddings.shape}")

# Create FAISS index

index = faiss.IndexFlatIP(dim)  # Using Inner Product (cosine similarity with normalized vectors)
faiss.normalize_L2(embeddings)  # normalize for cosine similarity
index.add(embeddings)
print(f"FAISS index has {index.ntotal} vectors")


# Save index

faiss.write_index(index, os.path.join(BASE_PATH, "vector_db.index"))
print("FAISS index saved")


# Save chunks

np.save(os.path.join(BASE_PATH, "chunks.npy"), chunks)
print(" Chunks saved")


# Save metadata (example: IDs)

metadata = [{"id": i} for i in range(len(chunks))]
np.save(os.path.join(BASE_PATH, "metadata.npy"), metadata)
print(" Metadata saved")
