"""
STEP 6: ReAct Agent with RAG Tool
===================================
Covers: @tool decorator, create_react_agent, ReAct loop,
        tool calling vs fixed chains

An Agent lets the LLM DECIDE which tools to use and when.
The ReAct pattern: Think → Act (call tool) → Observe (read result) → Repeat.

Unlike a fixed chain, the agent can call tools multiple times,
choose the best tool for each sub-question, and combine results.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

INDEX_NAME = "rag-bms"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Tools ─────────────────────────────────────────────────────────
# The docstring is what the LLM reads to decide WHEN to call this tool.
# Clear, specific docstrings → better tool selection by the agent.

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about LangChain concepts,
    RAG pipelines, agents, memory, LCEL, retrievers, and LangGraph.
    Use this whenever you need factual information to answer a question."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    results = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        results.append(f"[Chunk {i} | {src}]\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def compare_concepts(concept_a: str, concept_b: str) -> str:
    """Compare two LangChain concepts by searching for each separately.
    Use this when the user asks about differences or comparisons between concepts."""
    docs_a = retriever.invoke(concept_a)
    docs_b = retriever.invoke(concept_b)
    context_a = "\n".join(d.page_content for d in docs_a[:2])
    context_b = "\n".join(d.page_content for d in docs_b[:2])
    return f"=== {concept_a} ===\n{context_a}\n\n=== {concept_b} ===\n{context_b}"


# ── Agent ─────────────────────────────────────────────────────────
# create_react_agent from langgraph.prebuilt implements the ReAct loop:
#   1. LLM decides which tool to call (or whether to answer directly)
#   2. Tool executes, result added to message history
#   3. LLM sees the result and decides next action
#   4. Repeat until LLM produces a final answer
agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base, compare_concepts],
    prompt=SystemMessage(
        content="You are a LangChain expert. Always search the knowledge base "
                "before answering. Be specific and cite what you found."
    ),
)


def ask(question: str) -> str:
    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: REACT AGENT WITH RAG TOOL")
    print("=" * 60)

    questions = [
        "What is LCEL and how does it differ from the old LLMChain?",
        "Compare MultiQueryRetriever vs ContextualCompressionRetriever",
        "How do I handle errors in a LangChain chain? Give me the LCEL approach.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("─" * 60)
