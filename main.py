import os
import re
import json
import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.environ.get("API_KEY")
LLM_URL = os.environ.get("LLM_URL")

EMBEDDER_MODEL = "BAAI/bge-m3"
LLM_MODEL = "Qwen/Qwen3-32B"

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
# 1. EMBEDDER MODEL
# ===========================================================

def load_embedding_model():
    model = SentenceTransformer(EMBEDDER_MODEL)
    return model
    

# ===========================================================
# 2. CHUNKING FUNCTIONS
# (1) parse_topics for the data/topics.txt file
# (2) recursive_split_text for general text splitting
# ===========================================================

# Function for splitting topics file
def parse_topics(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    topics = []
    chunks = re.split(r"^##\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        description = "\n".join(lines[1:]).strip()
        text = f"{title}: {description}"
        topics.append(text)
    return topics
    
# Function for splitting text recursively
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

    # Apply overlap and ensure chunks start after the separator
    final_chunks = []
    for i, chunk in enumerate(base_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            prev = base_chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            # Remove the leading separator from the chunk
            chunk = chunk.lstrip("# ")
            final_chunks.append(overlap_text + " " + chunk)

    print(f"Total chunks created: {len(final_chunks)}")
    return final_chunks


# ===========================================================
# 3. CREATE CHROMADB VECTOR STORE with EMBEDDING FUNCTION
# ===========================================================

def create_file_search_store():
    """
    Creates a ChromaDB in-memory store and returns:
        model: embedding model
        client: chroma client
        collection: chroma collection
    """
    model = load_embedding_model()
   
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"           # tell Chroma to use YOUR loaded model
    )

    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory="rag_store"))

    # create or load collection
    collection = client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"},   # cosine similarity
        embedding_function=embedding_fn
    )

    return model, client, collection


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
        chunks = parse_topics(filepath)
    elif ext == ".pdf":
        text = read_pdf_file(filepath)
        chunks = recursive_split_text(text, chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError("Unsupported file type. Use TXT or PDF.")

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

def call_llm(prompt, api_key, model_name=LLM_MODEL):
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
        user_query = input("\nUser query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant chunks
        retrieved_docs = search_chroma(user_query, model, collection)
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Retrieved Document {i+1} ---\n{doc}\n")
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
    upload_document_to_store("data/topics.txt", model, collection, chunk_size=500, overlap=100)
    # upload_document_to_store("data/hu-embeddings-infocommunications-2025-11-03.pdf", model, collection, chunk_size=500, overlap=100)
    query_rag_system(model, collection, api_key=API_KEY)

