import os
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Config

BASE_FOLDER = r"C:\Users\Admin\Desktop\curio\voice_assistence_python\backend"
VECTOR_DB_PATH = os.path.join(BASE_FOLDER, "vector_db")
SUBJECTS = ["chemistry", "biology"]

# Chapter-based keywords per subject (NCERT chapters)
SUBJECT_KEYWORDS = {
    "chemistry": [
        "Some Basic Concepts of Chemistry",
        "Structure of Atom",
        "Chemical Bonding",
        "States of Matter",
        "Thermodynamics"
    ],
    "biology": [
        "The Cell",
        "Tissues",
        "Diversity of the Living World",
        "Human Physiology",
        "Plant Physiology"
    ]
}

CHUNK_SIZE = 500  # characters per chunk

# Ensure vector DB folder exists
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# PDF text extraction

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# Filter relevant lines (based on chapters)

def filter_relevant_text(text, keywords):
    filtered_text = ""
    for line in text.split("\n"):
        for kw in keywords:
            if kw.lower() in line.lower():
                filtered_text += line + "\n"
                break
    return filtered_text


# Split text into chunks

def split_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Main pipeline

all_chunks = []
all_metadata = []

for subject in SUBJECTS:
    folder_path = os.path.join(BASE_FOLDER, subject)
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        continue

    keywords = SUBJECT_KEYWORDS[subject]
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f" Processing: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(" No text found in PDF!")
                continue

            relevant_text = filter_relevant_text(text, keywords)
            if not relevant_text.strip():
                print(" No relevant chapter headers found, using full PDF text instead")
                relevant_text = text

            chunks = split_text(relevant_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "subject": subject,
                    "pdf": filename
                })

print(f" Total chunks collected: {len(all_chunks)}")


# Load sentence-transformers model (offline)

model_path = r"C:\Users\Admin\Desktop\hugging_face\all-distilroberta-v1"
embed_model = SentenceTransformer(model_path)


# Generate embeddings (offline)

embeddings = []
for i, chunk in enumerate(all_chunks):
    emb = embed_model.encode(chunk)
    embeddings.append(emb)

    if (i + 1) % 10 == 0:
        print(f"🔹 Processed {i+1}/{len(all_chunks)} chunks")

# Create FAISS vector DB

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))


# Save index & metadata

faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "vector_db.index"))
np.save(os.path.join(VECTOR_DB_PATH, "metadata.npy"), all_metadata)
np.save(os.path.join(VECTOR_DB_PATH, "chunks.npy"), all_chunks)

print(f" Vector DB and metadata saved to: {VECTOR_DB_PATH}")
