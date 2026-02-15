# NCERT RAG Question Answering System
> A Retrieval-Augmented Generation (RAG) based Question Answering system built using FastAPI (Backend) and React.js (Frontend).
> This system processes NCERT Class 11 Chemistry and Biology textbooks and generates contextual answers using semantic search and Gemini API.

---

## Project Overview
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline to answer questions from NCERT Class 11 Chemistry and Biology textbooks.

The system:
- Extracts text from PDFs
- Splits text into chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in FAISS vector database
- Retrieves relevant chunks
- Uses Gemini API to generate contextual answers

If no relevant chunk is found, the system returns:

> "Sorry, I don't have information on this topic."

---

## System Architecture

User Query  
⬇  
Query Embedding  
⬇  
FAISS Similarity Search  
⬇  
Retrieve Relevant Chunks  
⬇  
Send Context + Query to Gemini API  
⬇  
Generate Final Answer  

---

## Tech Stack
### Backend
- FastAPI
- FAISS
- Sentence Transformers (HuggingFace)
- Gemini API
- PyPDF2 / PdfReader

### Frontend
- React.js
- Axios

### Machine Learning Concepts
- Vector Embeddings
- Cosine Similarity
- Dense Retrieval
- Retrieval-Augmented Generation (RAG)

---

## Installation Guide

### Clone the Repository

```bash
git clone https://github.com/Valagatan31/NCERT-RAG-Chatbot.git
cd NCERT-RAG-Chatbot
```


