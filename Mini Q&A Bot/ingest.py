# ingest.py
import os
import faiss
import pickle
from rag.embedder import Embedder

DATA_DIR = "C:/Users/Administrator/Desktop/Mini Q&A Bot/data"
INDEX_PATH = "vector_store/faiss_index"
CHUNKS_PATH = "vector_store/chunks.pkl"
CHUNK_SIZE = 300

def read_text_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size)]

def main():
    os.makedirs("vector_store", exist_ok=True)

    embedder = Embedder()
    all_texts = read_text_files(DATA_DIR)

    all_chunks = []
    for text in all_texts:
        all_chunks.extend(chunk_text(text))

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = embedder.embed_texts(all_chunks)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ FAISS index and chunks saved.")

if __name__ == "__main__":
    main()
