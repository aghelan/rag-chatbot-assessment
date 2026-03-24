# RAG-based FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers user questions from a custom FAQ knowledge base. The system retrieves the most relevant FAQ entries using FAISS vector search and multilingual sentence embeddings, then generates grounded answers using the Gemini 2.5 Flash API.

This project was built as part of a Data Science / ML assessment to demonstrate practical understanding of:
- document ingestion and preprocessing
- embeddings and vector retrieval
- prompt grounding
- LLM integration
- web-based deployment

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
```

## Project Structure

```text
rag-chatbot-assessment/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── faq_clean.txt
│
├── src/
│   ├── guardrails.py
│   ├── ingest.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   └── retriever.py
│
└── vector_store/     # generated locally after ingestion
```

## Setup Instructions

1. Clone the repository
```
git clone https://github.com/aghelan/rag-chatbot-assessment.git
cd rag-chatbot-assessment
```

2. Create a virtual environment (Windows)
```
python -m venv .venv
.venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```

4. Configure the Gemini API key

Create a .env file in the project root and add:
```
GEMINI_API_KEY=your_valid_gemini_api_key_here
```
Notes:
- A valid Gemini API key is required for answer generation.
- Do not commit your real API key to GitHub.
- For Streamlit Cloud deployment, the same key should be added in Streamlit app secrets.

5. Build the vector store

Run the ingestion script:
```
python src/ingest.py
```
This will generate:
```
vector_store/index.faiss

vector_store/metadata.pkl
```

6. Run the application
```
streamlit run app.py
```
After running the command, open the local URL shown in the terminal, typically:
```
http://localhost:8501
```

## Quick Execution Steps
Windows
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create .env and add:
```
GEMINI_API_KEY=your_valid_gemini_api_key_here
```
Then run:
```
python src/ingest.py
```
```
streamlit run app.py
```

## How the System Works

### 1. Knowledge Base Preparation

The original FAQ document was manually cleaned and converted into a structured text file:
- each FAQ is separated clearly
- each entry includes:
 - FAQ ID
 - Category
 - Question
 - Answer

### 2. Ingestion

The ingestion pipeline:
- loads faq_clean.txt
- splits it into FAQ-level chunks
- generates embeddings
- stores the vectors in a FAISS index
- stores metadata in a pickle file

### 3. Retrieval

When a user asks a question:
- the query is embedded using the same multilingual embedding model
- FAISS searches for the closest matching FAQ chunks
- top matching documents are returned

### 4. Answer Generation

The retrieved documents are passed into a grounded prompt and sent to Gemini 2.5 Flash using Python `requests`.

The model is instructed to:
- answer only from the provided context
- avoid hallucinating facts
- return a fallback response if the answer is unclear

### 5. User Interface
The chatbot is exposed through a simple Streamlit web interface where users can:
- type a question
- view the generated answer
- inspect retrieved source documents

## Example Queries

Try the following example questions:

- Bagaimana saya nak batalkan langganan TontonUp?
- Bagaimana saya nak menukar kata laluan?
- Boleh saya melanggan apabila saya di luar negara Malaysia?
- Kenapa saya masih nampak iklan walaupun sudah melanggan?
- Apakah program TV Tuisyen yang disediakan di platform Tonton?

## Guardrails

This project includes basic guardrails to improve safety and reliability.

### Prompt Filtering

The chatbot blocks clearly malicious or prompt-injection style inputs such as:
- attempts to reveal system prompts
- prompt bypass attempts
- malware or hacking related requests

### Retrieval Confidence Check

If the retrieved documents are too weak or not relevant enough, the chatbot returns a fallback response instead of forcing an answer.

This helps reduce unsupported or hallucinated outputs.

## Why Multilingual Embeddings Were Used

The FAQ knowledge base is written primarily in Malay.

During testing, an English-oriented embedding model produced weaker retrieval quality for Malay-language user queries. To improve semantic retrieval performance, the system was updated to use:

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```
This improved retrieval quality significantly for Malay FAQ matching.

##  Gemini API Note

The assessment document included a Gemini API key. During implementation and testing, that key returned an expired or invalid key response.

To ensure the project remains fully functional and reproducible, the application loads a valid Gemini API key from a local `.env` file instead.

This approach is also better from a security and software engineering perspective, since secrets should not be hardcoded into source files.

## Deployment Note

The deployed Streamlit app is configured to build the vector store automatically on first startup if the FAISS index files do not yet exist.

For deployed environments, the Gemini API key should be configured using Streamlit secrets.

## Repository Link

https://github.com/aghelan/rag-chatbot-assessment

## Deployed Application Link

https://rag-chatbot-assessment.streamlit.app/

## Author 
Aghelan Logasaravanan
