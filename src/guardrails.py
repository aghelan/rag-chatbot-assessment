from typing import Tuple


BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "show me your prompt",
    "bypass",
    "hack",
    "malware",
    "exploit",
    "jailbreak",
    "prompt injection"
]


def check_guardrails(user_query: str) -> Tuple[bool, str]:
    """
    Returns:
        (is_allowed, message)
    """
    query_lower = user_query.lower().strip()

    if not query_lower:
        return False, "Sila masukkan soalan yang sah."

    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return False, "Maaf, saya tidak dapat membantu dengan permintaan tersebut."

    return True, ""


def is_retrieval_confident(retrieved_docs, max_score_threshold: float = 12.0) -> bool:
    """
    For FAISS IndexFlatL2:
    lower score = better match

    If the top result score is too high, retrieval may be weak.
    """
    if not retrieved_docs:
        return False

    top_score = retrieved_docs[0].get("score", 9999.0)
    return top_score <= max_score_threshold