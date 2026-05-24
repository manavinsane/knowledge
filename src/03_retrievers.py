"""
STEP 3: Different Types of Retrievers
======================================
Covers: VectorStoreRetriever, MMR, MultiQueryRetriever,
        ContextualCompressionRetriever

Each retriever type solves a different retrieval problem.
Run 01_ingest.py before this.
"""
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from src.model.user import UserRole
from src.rag.access import metadata_filter_for_role

load_dotenv()

INDEX_NAME = "rag-bms"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

QUERY = "How does memory work in LangChain agents?"
DEMO_ROLE = UserRole.EMPLOYEE


@lru_cache
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


def show(label, docs):
    print(f"\n{'─'*50}")
    print(f"  {label}  →  {len(docs)} chunk(s) returned")
    print(f"{'─'*50}")
    for i, doc in enumerate(docs[:2], 1):
        src = doc.metadata.get("source", "?")
        print(f"  [{i}] ({src})\n      {doc.page_content[:120].strip()}...")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: DIFFERENT RETRIEVER TYPES")
    print(f"Query: {QUERY}")
    print("=" * 60)

    # ── 1. Basic Similarity Search ────────────────────────────────
    # Embeds the query, finds the k nearest vectors in Pinecone.
    # Simple, fast, but can return near-duplicate chunks.
    role_filter = metadata_filter_for_role(DEMO_ROLE)
    vectorstore = get_vectorstore()
    basic = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": role_filter})
    show("1. BASIC SIMILARITY (k=3)", basic.invoke(QUERY))

    # ── 2. MMR – Maximal Marginal Relevance ───────────────────────
    # fetch_k=10  →  retrieve 10 candidates
    # Then pick k=3 that are both relevant AND diverse (not clones).
    # lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7,
            "filter": role_filter,
        },
    )
    show("2. MMR – RELEVANCE + DIVERSITY (lambda=0.7)", mmr.invoke(QUERY))

    # ── 3. MultiQueryRetriever ────────────────────────────────────
    # The LLM rephrases your query in 3 different ways.
    # Runs retrieval for each phrasing, then deduplicates results.
    # Better for vague or ambiguous questions.
    multi = MultiQueryRetriever.from_llm(retriever=basic, llm=llm)
    show("3. MULTI-QUERY (3 rephrased queries, deduplicated)", multi.invoke(QUERY))

    # ── 4. ContextualCompressionRetriever ────────────────────────
    # Retrieve chunks as usual, then ask an LLM to extract ONLY the
    # parts that are relevant to the query. Removes noise.
    compressor = LLMChainExtractor.from_llm(llm)
    compressed = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=basic,
    )
    compressed_docs = compressed.invoke(QUERY)
    print(f"\n{'─'*50}")
    print(f"  4. CONTEXTUAL COMPRESSION  →  {len(compressed_docs)} compressed chunk(s)")
    print(f"{'─'*50}")
    for i, doc in enumerate(compressed_docs[:2], 1):
        print(f"  [{i}] COMPRESSED TEXT:\n      {doc.page_content[:200].strip()}")
