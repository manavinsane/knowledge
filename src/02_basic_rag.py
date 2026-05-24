"""
STEP 2: Basic RAG with LCEL (LangChain Expression Language)
============================================================
Covers: | pipe operator, RunnableParallel, RunnablePassthrough,
        StrOutputParser, as_retriever()

The | operator chains Runnables together — output of each step
becomes input of the next. This is LCEL.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

INDEX_NAME = "rag-bms"

# ── Load vector store (already populated by 01_ingest.py) ────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

# as_retriever() turns the vector store into a Retriever.
# A Retriever has a standard invoke(query) → list[Document] interface.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    """Join retrieved Document objects into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── RAG Prompt ───────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

# ── LCEL RAG Chain ───────────────────────────────────────────────
# Step-by-step breakdown:
#
#   RunnableParallel runs TWO things at the same time for the same input:
#     context  → retriever gets relevant docs, format_docs joins them
#     question → RunnablePassthrough passes the input string unchanged
#
#   The result is {"context": "...", "question": "..."} which feeds into:
#     prompt   → fills the template with context + question
#     llm      → calls the model, returns AIMessage
#     parser   → extracts just the string from AIMessage
#
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: BASIC RAG WITH LCEL")
    print("=" * 60)

    questions = [
        "you know who is the rulling party in west bengal right now?",
        "What are the different types of text splitters?",
        "How does an agent decide which tool to use?",
        "What is the difference between a vector store and a retriever?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print("-" * 60)
