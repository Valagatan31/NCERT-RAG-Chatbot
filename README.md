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
- Perfrom Quantization on whisper model 

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
### backend setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### create .env file 
```bash
OPENAI_API_KEY=your_key
HUGGINGFACE_TOKEN=your_token
```

### run backend
```bash
uvicorn app.main:app --reload
```

### frontend setup
```bash
cd frontend
npm install
npm start
```

## Key Features
- Context-aware answering using RAG
- Semantic similarity search with FAISS
- Clean REST API using FastAPI
- React-based UI
- Secure API key management
- Fallback response when no relevant data is found

## Future improvement 
- Add similarity threshold filtering
- Add metadata (chapter, page number)
- Implement evaluation metrics (Precision@K)
- Add hybrid search (BM25 + Dense Retrieval)
- Dockerize the application

