import os
import re
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/faq_clean.txt"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_faq_text(file_path: str) -> str:
    """Load the cleaned FAQ text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FAQ file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_faq_blocks(text: str) -> List[str]:
    """
    Split the FAQ text into individual FAQ blocks using [FAQ X] markers.
    Each FAQ block is treated as one chunk.
    """
    blocks = re.split(r"(?=\[FAQ \d+\])", text)
    blocks = [block.strip() for block in blocks if block.strip()]
    return blocks


def extract_metadata(block: str) -> Dict[str, str]:
    """
    Extract FAQ metadata from a single block.
    """
    faq_match = re.search(r"\[FAQ (\d+)\]", block)
    category_match = re.search(r"Category:\s*(.*)", block)
    question_match = re.search(r"Question:\s*(.*)", block)

    faq_id = faq_match.group(1).strip() if faq_match else "Unknown"
    category = category_match.group(1).strip() if category_match else "Unknown"
    question = question_match.group(1).strip() if question_match else "Unknown"

    return {
        "faq_id": faq_id,
        "category": category,
        "question": question,
        "text": block
    }


def build_embeddings(chunks: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Generate embeddings for all chunks using SentenceTransformer.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    return np.array(embeddings).astype("float32")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Create a FAISS index using L2 distance.
    """
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings array must be 2-dimensional.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def save_vector_store(index: faiss.IndexFlatL2, metadata: List[Dict[str, str]]) -> None:
    """
    Save FAISS index and metadata to disk.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


def main() -> None:
    print("Step 1: Loading FAQ text...")
    text = load_faq_text(DATA_PATH)

    print("Step 2: Splitting FAQ blocks...")
    blocks = split_faq_blocks(text)
    print(f"Total FAQ chunks found: {len(blocks)}")

    if not blocks:
        raise ValueError("No FAQ blocks found. Please check the format of faq_clean.txt")

    print("Step 3: Extracting metadata...")
    metadata = [extract_metadata(block) for block in blocks]

    print("Preview of extracted FAQ entries:")
    for item in metadata:
        print(f"- FAQ {item['faq_id']}: {item['question']} [{item['category']}]")

    chunks = [item["text"] for item in metadata]

    print("Step 4: Building embeddings...")
    embeddings = build_embeddings(chunks)

    print("Step 5: Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("Step 6: Saving vector store...")
    save_vector_store(index, metadata)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()