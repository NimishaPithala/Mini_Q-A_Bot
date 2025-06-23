# rag/retriever.py
import faiss
import pickle
import numpy as np
from rag.embedder import Embedder

class Retriever:
    def __init__(self, index_path="vector_store/faiss_index", chunks_path="vector_store/chunks.pkl"):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.embedder = Embedder()

    def retrieve(self, query, top_k=3):
        query_vector = self.embedder.embed_texts([query])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        return [self.chunks[i] for i in indices[0]]
