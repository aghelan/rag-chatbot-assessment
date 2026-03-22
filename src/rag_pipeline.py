from typing import Dict, Any

from retriever import FAQRetriever
from llm import generate_answer
from guardrails import check_guardrails, is_retrieval_confident


class RAGPipeline:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.retriever = FAQRetriever()

    def run(self, user_query: str) -> Dict[str, Any]:
        is_allowed, guardrail_message = check_guardrails(user_query)
        if not is_allowed:
            return {
                "query": user_query,
                "answer": guardrail_message,
                "retrieved_docs": []
            }

        retrieved_docs = self.retriever.retrieve(user_query, top_k=self.top_k)

        if not retrieved_docs:
            return {
                "query": user_query,
                "answer": "Saya tidak dapat menemui maklumat yang berkaitan dalam dokumen yang diberikan.",
                "retrieved_docs": []
            }

        if not is_retrieval_confident(retrieved_docs):
            return {
                "query": user_query,
                "answer": "Saya tidak dapat menemui jawapan yang jelas dalam dokumen yang diberikan.",
                "retrieved_docs": retrieved_docs
            }

        answer = generate_answer(user_query, retrieved_docs)

        return {
            "query": user_query,
            "answer": answer,
            "retrieved_docs": retrieved_docs
        }


if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=3)

    test_query = "Bagaimana saya nak batalkan langganan TontonUp?"
    result = pipeline.run(test_query)

    print("\nUser Query:")
    print(result["query"])

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nRetrieved Documents:")
    for i, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"\nDocument {i}")
        print(f"FAQ ID   : {doc['faq_id']}")
        print(f"Category : {doc['category']}")
        print(f"Question : {doc['question']}")
        print(f"Score    : {doc['score']}")