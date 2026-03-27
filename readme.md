# 🔍 RAG Pipeline — PDF Question Answering System

A complete **Retrieval-Augmented Generation (RAG)** pipeline built from scratch using Python, ChromaDB, and Ollama. Upload any PDF and ask questions about it using a local LLM — no OpenAI API key needed!

---

## 📌 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines:
- **Retrieval** → find relevant text from your documents
- **Generation** → use an LLM to answer based on that text

Instead of relying on the LLM's training data, it answers from **your own PDFs**.

---

## 🔄 Pipeline Flow

```
INDEXING PHASE
PDF ──► Text Extraction ──► Chunking ──► Embeddings ──► ChromaDB

QUERY PHASE
User Query ──► Embeddings ──► ChromaDB Similarity Search ──► Relevant Chunks ──► LLM ──► Answer
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| PDF Parsing | PyPDF |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| LLM | Ollama (Llama3.2) — runs locally |
| Alternative DB | FAISS |

---

## 📁 Project Structure

```
rag_pipeline/
├── ingest.py          # PDF → Embeddings → ChromaDB
├── query.py           # User Query → Search → LLM Answer
├── config.py          # Settings (model name, chunk size, etc.)
├── requirements.txt   # Python dependencies
├── pdfs/              # Place your PDF files here
└── chroma_db/         # Auto-created: ChromaDB storage (git ignored)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and setup Ollama
```bash
# Linux / Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows → Download from https://ollama.com

# Pull the model
ollama pull llama3.2

# Start Ollama server
ollama serve
```

---

## 🚀 How to Use

### Step 1 — Add your PDFs
```
Copy your PDF files into the pdfs/ folder
```

### Step 2 — Ingest PDFs into ChromaDB
```bash
python ingest.py
```
This will:
- Read all PDFs from the `pdfs/` folder
- Split them into chunks
- Generate embeddings
- Store everything in ChromaDB

### Step 3 — Start asking questions
```bash
python query.py
```

```
🔍 RAG Pipeline ready (Ollama)! Type 'exit' to quit.

You: What is machine learning?
Assistant: Machine learning is a subset of AI that enables systems to learn from data...

You: exit
```

---

## ⚙️ Configuration (`config.py`)

```python
COLLECTION_NAME = "rag_docs"          # ChromaDB collection name
EMBED_MODEL     = "all-MiniLM-L6-v2"  # Embedding model
CHUNK_SIZE      = 500                  # Characters per chunk
CHUNK_OVERLAP   = 50                   # Overlap between chunks
PDF_FOLDER      = "./pdfs"             # PDF source folder
```

---

## 🔄 Want to use OpenAI instead of Ollama?

In `query.py`, replace:
```python
# Ollama (local, free)
llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
model = "llama3.2"
```
With:
```python
# OpenAI (cloud, paid)
llm = OpenAI()   # set OPENAI_API_KEY in .env
model = "gpt-3.5-turbo"
```

---

## 🔄 Want to use FAISS instead of ChromaDB?

```python
# pip install faiss-cpu
import faiss, numpy as np, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Ingest
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype="float32"))
faiss.write_index(index, "faiss.index")
json.dump(chunks, open("chunks.json", "w"))

# Query
index = faiss.read_index("faiss.index")
q_vec = model.encode([query]).astype("float32")
_, indices = index.search(q_vec, k=3)
top_chunks = [chunks[i] for i in indices[0]]
```

---

## 📦 Dependencies

```
chromadb
sentence-transformers
pypdf
openai
langchain
faiss-cpu
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🙋 Author

Built as a learning project to understand RAG pipelines, vector databases, and local LLMs.

---

## 📄 License

MIT License — free to use and modify.