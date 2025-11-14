import os
import json
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.environ.get("API_KEY")
LLM_URL = os.environ.get("LLM_URL")

# ===========================================================
# 1. EMBEDDINGS
# ===========================================================

def load_embedding_model():
    # BGE-M3 via SentenceTransformers
    return SentenceTransformer("BAAI/bge-m3")


# ===========================================================
# 2. RECURSIVE CHUNKER WITH OVERLAP
# ===========================================================

def recursive_split_text(text, chunk_size=500, overlap=100):
    """
    Splits text recursively into overlapping chunks.
    Similar to LangChain's RecursiveCharacterTextSplitter.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split_recursively(text, sep_index=0):
        if len(text) <= chunk_size or sep_index >= len(separators):
            return [text]

        sep = separators[sep_index]
        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += part + sep
            else:
                if current:
                    chunks.append(current.strip())
                current = part + sep

        if current:
            chunks.append(current.strip())

        # If still too large: keep splitting with next separator
        result = []
        for c in chunks:
            if len(c) > chunk_size:
                result.extend(split_recursively(c, sep_index + 1))
            else:
                result.append(c)

        return result

    base_chunks = split_recursively(text)

    # Apply overlap
    final_chunks = []
    for chunk in base_chunks:
        if not final_chunks:
            final_chunks.append(chunk)
        else:
            prev = final_chunks[-1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            final_chunks.append(overlap_text + " " + chunk)

    return final_chunks


# ===========================================================
# 3. CREATE CHROMADB VECTOR STORE
# ===========================================================

def create_file_search_store():
    """
    Creates a ChromaDB in-memory store and returns:
        model: embedding model
        client: chroma client
        collection: chroma collection
    """
    model = load_embedding_model()

    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory="rag_store"))

    # create or load collection
    collection = client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    return model, client, collection


# ===========================================================
# FILE READERS
# ===========================================================

def read_txt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf_file(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ===========================================================
# 4. UPLOAD DOCUMENT TO CHROMADB
# ===========================================================

def upload_document_to_store(filepath, model, collection,
                             chunk_size=500, overlap=100):
    """
    Loads a TXT or PDF, recursively chunks it, embeds it,
    and stores chunks in ChromaDB.
    """

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        text = read_txt_file(filepath)
    elif ext == ".pdf":
        text = read_pdf_file(filepath)
    else:
        raise ValueError("Unsupported file type. Use TXT or PDF.")

    chunks = recursive_split_text(text, chunk_size=chunk_size, overlap=overlap)

    # generate IDs
    ids = [f"{os.path.basename(filepath)}_{i}" for i in range(len(chunks))]

    # embed chunks
    embeddings = model.encode(chunks).tolist()

    # store
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(f"Uploaded {len(chunks)} chunks to ChromaDB.")


# ===========================================================
# 5. QUERY CHROMADB
# ===========================================================

def search_chroma(query, model, collection, k=3):
    q_emb = model.encode([query]).tolist()[0]

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return result["documents"][0]


# ===========================================================
# 6. LLM CALL
# ===========================================================

def call_llm(prompt, api_key, model_name="Qwen/Qwen3-32B"):
    url = LLM_URL

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 1000,
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ===========================================================
# 7. QUERY RAG SYSTEM LOOP
# ===========================================================

def query_rag_system(model, collection, api_key):
    """
    Interactive RAG loop. Type 'exit' to stop.
    """

    print("RAG System ready. Type your query (or 'exit'): ")

    while True:
        user_query = input("\nUser: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks
        retrieved_docs = search_chroma(user_query, model, collection)

        context = "\n\n--- Retrieved Context ---\n" + "\n".join(retrieved_docs)

        prompt = f"""
Use ONLY the following context to answer the question.
If the answer is not in the context, say you don't know.

{context}

User question: {user_query}
"""

        response = call_llm(prompt, api_key)
        print("\nAssistant:", response)


# ===========================================================
# Example Usage
# ===========================================================

if __name__ == "__main__":
    model, client, collection = create_file_search_store()

    # upload_document_to_store("data/topics.txt", model, collection)
    
    query_rag_system(model, collection, api_key=API_KEY)
    pass
