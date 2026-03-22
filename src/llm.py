import os
from typing import List, Dict

import requests
from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:
    st = None


load_dotenv()


def get_gemini_api_key() -> str:
    """
    Load Gemini API key from:
    1. Local .env (for local development)
    2. Streamlit secrets (for deployed app)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    if st is not None:
        try:
            return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    return ""


def build_context(retrieved_docs: List[Dict]) -> str:
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[Document {i}]\n"
            f"FAQ ID: {doc.get('faq_id', 'Unknown')}\n"
            f"Category: {doc.get('category', 'Unknown')}\n"
            f"Question: {doc.get('question', 'Unknown')}\n"
            f"Content:\n{doc.get('text', '')}\n"
        )

    return "\n".join(context_parts)


def build_prompt(user_query: str, context: str) -> str:
    return f"""
You are a helpful FAQ assistant.

Answer the user's question using only the provided context.
If the answer is not clearly available in the context, say:
"Saya tidak dapat menemui jawapan yang jelas dalam dokumen yang diberikan."

Rules:
- Do not make up facts.
- Do not use outside knowledge.
- Answer clearly and naturally.
- If steps are available, present them in order.
- Keep the answer concise but complete.

Context:
{context}

User Question:
{user_query}
""".strip()


def generate_answer(user_query: str, retrieved_docs: List[Dict]) -> str:
    gemini_api_key = get_gemini_api_key()

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please check your Streamlit secrets or .env file.")

    gemini_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
    )

    context = build_context(retrieved_docs)
    prompt = build_prompt(user_query, context)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(gemini_url, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        error_text = response.text

        if response.status_code == 400:
            if "API key not valid" in error_text:
                raise RuntimeError("Gemini API key is invalid. Please check your Streamlit secret or .env file.")
            if "API key expired" in error_text:
                raise RuntimeError("Gemini API key has expired. Please replace it with a valid key.")

        raise RuntimeError(
            f"Gemini API request failed.\nStatus Code: {response.status_code}\nResponse: {error_text}"
        )

    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response format: {result}")