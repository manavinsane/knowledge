"""
STEP 7: LangGraph — Stateful, Graph-based RAG
==============================================
Covers: StateGraph, TypedDict state, nodes, edges,
        conditional routing, MemorySaver checkpointer, thread_id

LangGraph models your app as a directed graph:
  Nodes  = Python functions that read/write state
  Edges  = connections between nodes (fixed or conditional)
  State  = TypedDict passed between every node

Checkpointer + thread_id = automatic conversation memory.
Each call with the same thread_id picks up where it left off.
"""
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

load_dotenv()

INDEX_NAME = "rag-bms"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── State Schema ─────────────────────────────────────────────────
# Every node receives the full state and returns a dict of updates.
# add_messages is a reducer: instead of replacing the list, it appends.
# Without add_messages, each node would overwrite the whole history.
class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str          # retrieved document text
    should_retrieve: bool  # routing decision


# ── Node 1: Routing ───────────────────────────────────────────────
# Decides whether the current message needs retrieval.
# Simple questions like "thanks!" should skip retrieval entirely.
def decide(state: RAGState) -> dict:
    last = state["messages"][-1].content.lower()
    retrieval_signals = ["what", "how", "why", "explain", "describe", "?", "tell me"]
    needs = any(signal in last for signal in retrieval_signals)
    return {"should_retrieve": needs}


# ── Node 2: Retrieval ─────────────────────────────────────────────
def retrieve(state: RAGState) -> dict:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return {"context": context}


# ── Node 3: Generation ────────────────────────────────────────────
def generate(state: RAGState) -> dict:
    context = state.get("context", "")
    system_content = "You are a helpful LangChain assistant."
    if context:
        system_content += f"\n\nAnswer using ONLY this context:\n{context}"

    # Build the message list for the LLM call
    prompt_messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(prompt_messages)
    return {"messages": [AIMessage(content=response.content)]}


# ── Conditional Routing Function ──────────────────────────────────
# This function is called after the "decide" node.
# It returns the name of the next node to go to.
def route(state: RAGState) -> str:
    return "retrieve" if state["should_retrieve"] else "generate"


# ── Build the Graph ───────────────────────────────────────────────
builder = StateGraph(RAGState)

builder.add_node("decide", decide)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "decide")
builder.add_conditional_edges(
    "decide",
    route,
    {"retrieve": "retrieve", "generate": "generate"},
)
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# MemorySaver stores state in memory (use PostgresSaver in production).
# With a checkpointer, every step's state is saved automatically.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: LANGGRAPH STATEFUL RAG")
    print("=" * 60)

    # thread_id scopes the conversation — same ID = same memory.
    config = {"configurable": {"thread_id": "demo-session-1"}}

    conversation = [
        "What is LCEL in LangChain?",
        "How is it different from the old way of chaining?",  # uses graph state
        "What retriever types exist?",
        "Thank you!",                                         # should skip retrieval
    ]

    for message in conversation:
        print(f"\nHuman: {message}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        response = result["messages"][-1].content
        retrieved = result.get("should_retrieve", False)
        print(f"AI:    {response}")
        print(f"       [retrieved docs: {retrieved}]")
        print("─" * 60)

    # ── Time Travel ───────────────────────────────────────────────
    print("\n\nGRAPH STATE HISTORY (time travel):")
    print("Each step below is a saved checkpoint you can replay from.\n")
    for i, step in enumerate(graph.get_state_history(config)):
        last_msg = step.values.get("messages", [])
        if last_msg:
            print(f"  Step {i}: {last_msg[-1].__class__.__name__}: "
                  f"{last_msg[-1].content[:60]}...")
