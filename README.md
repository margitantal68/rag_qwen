# 📚 RAG System with ChromaDB and LLM
This project is a Retrieval-Augmented Generation (RAG) system that uses ChromaDB for storing and retrieving document chunks and an LLM (Large Language Model) for generating answers based on the retrieved information.

## ✅ Features
- **Recursive Text Splitter:** Splits text into overlapping chunks for better context retention.
- **ChromaDB Integration:** Stores and retrieves document embeddings using ChromaDB with cosine similarity.
- **Supports PDF and TXT:** Reads and processes both .txt and .pdf documents.
- **Interactive Query Loop:** Asks the LLM for answers based on the retrieved chunks.
- **Custom Embedding Model:** Uses BAAI/bge-m3 for generating document embeddings.
- **LLM API Integration:** Communicates with a custom LLM API to generate responses.

## Requirements
- Python 3.11

## 📦 Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag_qwen
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  
   # On Windows use `.venv\Scripts\activate`
   ```

3. Create a `.env` file based on the `.env.example`:
   ```bash
   cp .env.example .env
   ```

4. Add your LLM_URI and API key to the `.env` file:
   ```
   API_KEY=YOUR_API_KEY_HERE
   LLM_URL=YOUR_LLM_URL_HERE
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Usage Notes
- Make sure your LLM API is running and accessible at the endpoint specified in call_llm().
- The default model used is Qwen/Qwen3-32B, but you can change it to any model supported by the API.
- Run the main application:
   ```bash
   python main.py
   ```

## Data Files
- `data/topics.txt`: Provides detailed information about various topics.
- `data/cs_qa.csv`: Contains questions and answers related to the Clearservice company.

