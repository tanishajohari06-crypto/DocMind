# DocMind
# DocMind: Multi-LLM RAG-Based Document Intelligence System

## Abstract

DocMind is a Retrieval-Augmented Generation (RAG) based system designed to enable intelligent interaction with unstructured documents. It combines semantic search with multiple Large Language Models (LLMs) to generate context-aware responses. The system retrieves relevant document segments using vector embeddings and compares outputs from different LLMs, improving answer accuracy and reliability.

---

## Keywords

RAG, Large Language Models, Semantic Search, FAISS, Document Analysis, AI

---

## Overview

DocMind allows users to upload PDF documents and ask natural language questions. The system processes the document, retrieves relevant content using embeddings, and generates responses using multiple LLMs for comparison.

---

## Features

* PDF document upload and processing
* Semantic search using embeddings
* Multi-LLM response comparison
* Fast retrieval with FAISS
* Interactive Q&A interface

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Tools:** FAISS, Sentence Transformers, PyMuPDF, NumPy, Groq API

---

## How It Works

1. Upload PDF
2. Extract and split text
3. Generate embeddings
4. Store using FAISS
5. Retrieve relevant content
6. Generate answers using multiple LLMs

---

## Setup Instructions

```bash
git clone https://github.com/tanishajohari06-crypto/DocMind.git
cd DocMind
pip install -r requirements.txt
streamlit run app.py
```

---


---

## Conclusion

DocMind demonstrates the effectiveness of combining RAG architecture with multiple LLMs to improve document understanding and answer accuracy.
