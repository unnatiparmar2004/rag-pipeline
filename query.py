import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from config import COLLECTION_NAME,EMBED_MODEL

# ChromaDB connection
client=chromadb.PersistentClient(path="./chroma_db")
embed_fn=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection=client.get_or_create_collection(name=COLLECTION_NAME,embedding_function=embed_fn)

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    results = collection.query(query_texts=[query],n_results=top_k)
    return results["documents"][0]

def ask(query: str) -> str:
    chunks  = retrieve(query)
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

CONTEXT:
{context}

QUESTION: {query}
ANSWER:"""

    response = llm.chat.completions.create(
        model    = "llama3.2",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("RAG Pipeline ready (Ollama)")
    while True:
        q = input("You: ").strip()
        if q.lower() == "exit":
            break
        if q:
            print(f"\nAssistant: {ask(q)}\n")