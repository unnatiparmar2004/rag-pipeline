import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from config import COLLECTION_NAME,EMBED_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,PDF_FOLDER

client=chromadb.PersistentClient(path="./chroma_db")
embed_fn=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection=client.get_or_create_collection(name=COLLECTION_NAME,embedding_function=embed_fn)

def extract_text_from_pdf(pdf_path: str) -> str:
    reader=PdfReader(pdf_path)
    text  =""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]

def is_already_ingested(pdf_name: str) -> bool:
    results = collection.get(where={"source_file": pdf_name})
    return len(results["ids"]) > 0

def ingest_pdf(pdf_path: str):
    pdf_name = os.path.basename(pdf_path)

    if is_already_ingested(pdf_name):
        print(f"Skipping (already ingested): {pdf_name}")
        return

    print(f"Ingesting: {pdf_path}")
    text=extract_text_from_pdf(pdf_path)
    chunks=chunk_text(text)

    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas=[{"source": pdf_path, "source_file": pdf_name, "chunk": i} for i in range(len(chunks))]

    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks")

if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    pdfs=[f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdfs:
        print("No PDFs found in ./pdfs folder!")
    else:
        for pdf in pdfs:
            ingest_pdf(os.path.join(PDF_FOLDER, pdf))
        print(f"\nDone! Total docs in DB: {collection.count()}")