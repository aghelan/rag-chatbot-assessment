import os
from typing import List, Dict

import requests
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"


def build_context(retrieved_docs: List[Dict]) -> str:
    """
    Combine retrieved FAQ chunks into a single context string.
    """
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
    """
    Build a grounded RAG prompt for Gemini.
    """
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
    """
    Generate an answer from Gemini using retrieved context.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please check your .env file.")

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

    response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API request failed.\nStatus Code: {response.status_code}\nResponse: {response.text}"
        )

    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response format: {result}")


if __name__ == "__main__":
    sample_docs = [
        {
            "faq_id": "3",
            "category": "Subscription Cancellation",
            "question": "Bagaimana saya nak membatalkan langganan bulanan TontonUp saya?",
            "text": """[FAQ 3]
Category: Subscription Cancellation
Question: Bagaimana saya nak membatalkan langganan bulanan TontonUp saya?
Answer:
Untuk makluman, anda boleh membatalkan pembayaran berulang automatik anda pada Profil Tonton anda."""
        }
    ]

    test_query = "Bagaimana saya nak batalkan langganan TontonUp?"
    answer = generate_answer(test_query, sample_docs)

    print("\nGemini Answer:\n")
    print(answer)