"""
build_index.py
Builds a local vector index (ChromaDB) from the banking knowledge base.
Run this once before starting the chatbot.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

KB_PATH = "knowledge_base.json"
DB_PATH = "./chroma_store"
COLLECTION_NAME = "banking_faq"

def main():
    print("Loading knowledge base...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"Loaded {len(kb)} FAQ entries.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Reset collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids = [item["id"] for item in kb]
    documents = [f"{item['question']} {item['answer']}" for item in kb]
    metadatas = [
        {"question": item["question"], "answer": item["answer"], "category": item["category"]}
        for item in kb
    ]

    print("Generating embeddings...")
    embeddings = model.encode(documents).tolist()

    print("Storing in ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Index built successfully with {len(kb)} entries at {DB_PATH}")

if __name__ == "__main__":
    main()
