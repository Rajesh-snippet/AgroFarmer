"""
AgroFarmer - RAG Service
Loads rice disease PDFs → chunks → embeds → stores in ChromaDB
Exposes a retrieve(query) function used by /recommend endpoint
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DOCS_DIR    = "knowledge_base/docs"
CHROMA_DIR  = "knowledge_base/chroma_db"

# ── Embedding model (runs locally, no API key needed) ────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight, good for semantic search

# ── Globals (loaded once) ────────────────────────────────────────────────────
_vectorstore = None


def _load_pdfs() -> list:
    """Load all PDFs from DOCS_DIR and return list of LangChain documents."""
    documents = []
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")

    print(f"📄 Loading {len(pdf_files)} PDFs...")
    for pdf_file in pdf_files:
        path = os.path.join(DOCS_DIR, pdf_file)
        try:
            loader = PyPDFLoader(path)
            docs   = loader.load()
            # Tag each chunk with source filename for traceability
            for doc in docs:
                doc.metadata["source"] = pdf_file
            documents.extend(docs)
            print(f"   ✅ {pdf_file} — {len(docs)} pages loaded")
        except Exception as e:
            print(f"   ❌ Failed to load {pdf_file}: {e}")

    print(f"📚 Total pages loaded: {len(documents)}")
    return documents


def _chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # ~700 chars per chunk — good balance for agricultural text
        chunk_overlap=100,     # overlap prevents losing context at chunk boundaries
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Total chunks created: {len(chunks)}")
    return chunks


def build_vectorstore() -> Chroma:
    """
    Build ChromaDB vectorstore from PDFs.
    Call this once to create the DB — subsequent runs load from disk.
    """
    global _vectorstore

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if GPU available locally
    )

    # If ChromaDB already exists on disk, load it (skip re-embedding)
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("📦 Loading existing ChromaDB from disk...")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        print(f"✅ ChromaDB loaded — {_vectorstore._collection.count()} chunks indexed")
        return _vectorstore

    # First run — load PDFs, chunk, embed, persist
    print("🔨 Building ChromaDB from scratch...")
    documents = _load_pdfs()
    chunks    = _chunk_documents(documents)

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ ChromaDB built and saved — {_vectorstore._collection.count()} chunks indexed")
    return _vectorstore


def get_vectorstore() -> Chroma:
    """Return the vectorstore, building it if not already loaded."""
    global _vectorstore
    if _vectorstore is None:
        build_vectorstore()
    return _vectorstore


def retrieve(query: str, k: int = 4) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a given query.

    Args:
        query: natural language query e.g. "How to treat rice leaf blast?"
        k:     number of chunks to retrieve (default 4)

    Returns:
        List of dicts with 'content' and 'source' keys
    """
    vs      = get_vectorstore()
    results = vs.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "unknown"),
        }
        for doc in results
    ]


# ── Quick test (run this file directly to verify RAG is working) ─────────────
if __name__ == "__main__":
    print("=== AgroFarmer RAG Service Test ===\n")

    # Build / load vectorstore
    build_vectorstore()

    # Test retrieval for each disease
    test_queries = [
        "How to treat rice leaf blast disease?",
        "What causes bacterial leaf blight in rice?",
        "How to control sheath blight in paddy?",
        "Rice tungro virus symptoms and management",
        "Brown spot disease treatment in rice",
    ]

    print("\n=== Retrieval Test ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retrieve(query, k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] Source: {r['source']}")
            print(f"       Preview: {r['content'][:150].strip()}...")