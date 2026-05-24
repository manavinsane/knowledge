"""
STEP 1: Ingestion Pipeline
==========================
Covers: Document Loaders, Text Splitters, Embeddings, Pinecone Vector Store

Flow: Load files → Split into chunks → Embed → Store in Pinecone

Run this FIRST before any other script.
"""
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "rag-bms"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # must match the model's output dimension


def ingest():
    print("=" * 60)
    print("STEP 1: INGESTION PIPELINE")
    print("=" * 60)

    # ── 1. Load Documents ─────────────────────────────────────────
    # DirectoryLoader scans a folder; TextLoader reads each .txt file.
    # Each file becomes one or more Document objects:
    #   doc.page_content  → the raw text
    #   doc.metadata      → {"source": "documents/file.txt", ...}
    print("\n[1/4] Loading documents from ./documents/ ...")
    loader = DirectoryLoader(
        "./documents",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"      Loaded {len(docs)} file(s)")
    for doc in docs:
        print(f"        - {doc.metadata['source']}  ({len(doc.page_content):,} chars)")

    # ── 2. Split Documents ────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries delimiters in order:
    #   "\n\n" (paragraphs) → "\n" (lines) → ". " (sentences) → " " → ""
    # chunk_overlap ensures context isn't lost at chunk boundaries.
    print("\n[2/4] Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,        # ~150 words per chunk
        chunk_overlap=100,     # 100 chars shared between adjacent chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(docs)
    print(f"      {len(docs)} file(s) → {len(splits)} chunks")
    print(f"      Sample chunk:\n        {splits[0].page_content[:150].strip()}...")

    # ── 3. Set up Pinecone Index ──────────────────────────────────
    print("\n[3/4] Setting up Pinecone index ...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"      Creating index '{INDEX_NAME}' (first time only) ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free tier
        )
        # Wait until index is ready
        while not pc.describe_index(INDEX_NAME).status.get("ready", False):
            time.sleep(1)
        print("      Index ready.")
    else:
        print(f"      Index '{INDEX_NAME}' already exists.")

    # ── 4. Embed + Upsert ────────────────────────────────────────
    # OpenAIEmbeddings converts each chunk to a 1536-dim vector.
    # PineconeVectorStore.from_documents embeds every chunk and upserts
    # it into Pinecone in batches.
    print("\n[4/4] Embedding chunks and upserting into Pinecone ...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"      Upserted {len(splits)} chunks into Pinecone index '{INDEX_NAME}'")

    print("\nDone! Now run 02_basic_rag.py to query your knowledge base.")
    return vectorstore


if __name__ == "__main__":
    ingest()
