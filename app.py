import os
import sys
import subprocess

import streamlit as st

# Ensure src folder is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag_pipeline import RAGPipeline


VECTOR_INDEX_PATH = os.path.join("vector_store", "index.faiss")
METADATA_PATH = os.path.join("vector_store", "metadata.pkl")


def ensure_vector_store():
    """
    Build the vector store automatically if it does not exist.
    Useful for first-time deployment environments.
    """
    if not os.path.exists(VECTOR_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        with st.spinner("Preparing vector store for first-time startup..."):
            result = subprocess.run(
                [sys.executable, "src/ingest.py"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to build vector store.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                )


st.set_page_config(
    page_title="RAG FAQ Chatbot",
    page_icon="💬",
    layout="centered"
)


@st.cache_resource
def load_pipeline():
    ensure_vector_store()
    return RAGPipeline(top_k=2)


pipeline = load_pipeline()

st.title("💬 RAG-based FAQ Chatbot")
st.markdown(
    """
    This chatbot answers user questions from a custom FAQ knowledge base using:
    - Retrieval-Augmented Generation (RAG)
    - FAISS vector search
    - SentenceTransformers multilingual embeddings
    - Gemini 2.5 Flash API
    """
)

st.caption("Answers are generated based only on the uploaded FAQ knowledge base.")

st.divider()

user_query = st.text_input(
    "Enter your question:",
    placeholder="e.g. Bagaimana saya nak batalkan langganan TontonUp?"
)

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            try:
                result = pipeline.run(user_query)

                st.subheader("Answer")
                st.markdown(result["answer"])

                with st.expander("Retrieved Source Documents"):
                    if result["retrieved_docs"]:
                        for i, doc in enumerate(result["retrieved_docs"], start=1):
                            st.markdown(f"**Document {i}**")
                            st.write(f"**FAQ ID:** {doc['faq_id']}")
                            st.write(f"**Category:** {doc['category']}")
                            st.write(f"**Question:** {doc['question']}")
                            st.write(f"**Score:** {doc['score']:.4f}")
                            st.code(doc["text"], language="text")
                            st.markdown("---")
                    else:
                        st.write("No supporting documents retrieved.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")