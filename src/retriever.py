import os
import pickle
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FAQRetriever:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.model = None
        self._load_assets()

    def _load_assets(self) -> None:
        """Load FAISS index, metadata, and embedding model."""
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Retriever is ready.")

    def embed_query(self, query: str) -> np.ndarray:
        """Convert a user query into an embedding."""
        query_embedding = self.model.encode([query])
        return np.array(query_embedding).astype("float32")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the top_k most relevant FAQ chunks for a query.
        Returns metadata along with FAISS distance score.
        """
        if not query.strip():
            return []

        query_vector = self.embed_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            result = self.metadata[idx].copy()
            result["score"] = float(distances[0][rank])
            results.append(result)

        return results


if __name__ == "__main__":
    retriever = FAQRetriever()

    test_query = "Bagaimana cara batalkan langganan TontonUp?"
    results = retriever.retrieve(test_query, top_k=3)

    print("\nTop retrieval results:")
    for i, item in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"FAQ ID   : {item['faq_id']}")
        print(f"Category : {item['category']}")
        print(f"Question : {item['question']}")
        print(f"Score    : {item['score']}")
        print(f"Text     :\n{item['text']}")