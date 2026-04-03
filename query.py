from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from config import (
    COLLECTION_NAME, EMBED_MODEL, CHROMA_PATH,
    OLLAMA_BASE_URL, LLM_MODEL
)

# ── Embeddings & Vector Store ────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# ── Retriever ────────────────────────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ── LLM (Ollama via LangChain) ───────────────────────────────────────────────
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2
)

# ── Custom Prompt ────────────────────────────────────────────────────────────
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't know."

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
)

# ── Helper: format retrieved docs into a single string ───────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── LCEL Chain (replaces RetrievalQA) ────────────────────────────────────────
# RunnableParallel fetches both the raw docs (for sources) and formatted context
retrieve_parallel = RunnableParallel(
    question=RunnablePassthrough(),
    docs=retriever,
    context=retriever | format_docs
)

qa_chain = (
    retrieve_parallel
    | {
        "answer": prompt_template | llm | StrOutputParser(),
        "source_documents": lambda x: x["docs"]
    }
)

# ── Ask Function ─────────────────────────────────────────────────────────────
def ask(query: str) -> str:
    result  = qa_chain.invoke(query)
    answer  = result["answer"]
    sources = result["source_documents"]

    # Print source files used (helpful for debugging)
    source_files = list({doc.metadata.get("source_file", "unknown") for doc in sources})
    print(f"  [Sources used: {', '.join(source_files)}]")

    return answer

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RAG Pipeline ready (LangChain + Ollama)")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() == "exit":
            break
        if q:
            print(f"\nAssistant: {ask(q)}\n")