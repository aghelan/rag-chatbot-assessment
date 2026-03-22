# RAG-based FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers user questions from a custom FAQ knowledge base. The system retrieves the most relevant FAQ entries using FAISS vector search and multilingual sentence embeddings, then generates grounded answers using the Gemini 2.5 Flash API.

This project was built as part of a Data Science / ML assessment to demonstrate practical understanding of:
- document ingestion and preprocessing
- embeddings and vector retrieval
- prompt grounding
- LLM integration
- simple deployment through a web interface

---

## Overview

This chatbot takes a user query, retrieves the most relevant FAQ entries from a preprocessed FAQ knowledge base, and generates an answer using a hosted LLM.

The project uses:
- **FAISS** for vector similarity search
- **SentenceTransformers** multilingual embeddings for better Malay-language retrieval
- **Gemini 2.5 Flash API** for answer generation
- **Streamlit** for the user interface

The chatbot is designed to answer only from the provided FAQ knowledge base and includes simple guardrails for harmful or irrelevant prompts.

---

## Features

- Custom FAQ ingestion pipeline
- Semantic retrieval using FAISS
- Multilingual embedding model for Malay FAQ matching
- Gemini 2.5 Flash integration using Python `requests`
- Streamlit-based chatbot UI
- Retrieved source documents shown in the interface for transparency
- Basic guardrails for malicious prompts
- Low-confidence retrieval fallback to reduce unsupported answers

---

## Project Architecture

```text
FAQ Document
   ↓
Manual cleaning and structuring
   ↓
FAQ-level chunking
   ↓
SentenceTransformers embeddings
   ↓
FAISS vector store
   ↓
Retriever
   ↓
Gemini 2.5 Flash API
   ↓
Streamlit Chatbot UI

---

## Author 

Aghelan Logasaravanan
