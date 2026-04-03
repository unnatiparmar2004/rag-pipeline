import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    COLLECTION_NAME, EMBED_MODEL, CHUNK_SIZE,
    CHUNK_OVERLAP, PDF_FOLDER, CHROMA_PATH
)

# ── Embeddings & Vector Store ────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# ── Text Splitter ────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def is_already_ingested(pdf_name: str) -> bool:
    """Check if a PDF has already been ingested by looking for its metadata."""
    results = vectorstore.get(where={"source_file": pdf_name})
    return len(results["ids"]) > 0

def ingest_pdf(pdf_path: str):
    pdf_name = os.path.basename(pdf_path)

    if is_already_ingested(pdf_name):
        print(f"Skipping (already ingested): {pdf_name}")
        return

    print(f"Ingesting: {pdf_path}")

    # LangChain PDF loader — returns list of Document objects (one per page)
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    # Split pages into smaller chunks
    chunks = splitter.split_documents(pages)

    # Attach custom metadata so we can filter later
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = pdf_name
        chunk.metadata["chunk"]       = i

    # Add to ChromaDB via LangChain
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks from '{pdf_name}'")

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdfs:
        print("No PDFs found in ./pdfs folder!")
    else:
        for pdf in pdfs:
            ingest_pdf(os.path.join(PDF_FOLDER, pdf))

        total = vectorstore._collection.count()
        print(f"\nTotal docs in DB: {total}")