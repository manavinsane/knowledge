"""
STEP 4: Conversational RAG with Chat History
=============================================
Covers: create_history_aware_retriever, create_retrieval_chain,
        MessagesPlaceholder, chat_history state management

Problem: follow-up questions like "tell me more about that" don't work
with plain RAG because the retriever never sees the previous messages.

Solution: first rewrite the question into a standalone question using
the chat history, THEN retrieve and answer.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

INDEX_NAME = "rag-bms"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Chain 1: Rewrite the question as a standalone question ────────
# Without this, "tell me more" has no context for the retriever.
# With this, "tell me more" → "What else should I know about LCEL?"
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest question, rewrite the
question as a complete standalone question that makes sense without the history.
Return ONLY the rewritten question — no explanation."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
contextualize_chain = contextualize_prompt | llm | StrOutputParser()


def get_retriever_input(state: dict) -> list:
    """Rewrite the question if there is history; otherwise use it as-is."""
    chat_history = state.get("chat_history", [])
    if chat_history:
        standalone = contextualize_chain.invoke(state)
        return retriever.invoke(standalone)
    return retriever.invoke(state["input"])


# ── Chain 2: Answer using retrieved context + history ─────────────
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say so.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

conversational_rag = (
    RunnablePassthrough.assign(context=get_retriever_input | format_docs)
    | answer_prompt
    | llm
    | StrOutputParser()
)


def chat(question: str, history: list) -> str:
    return conversational_rag.invoke({"input": question, "chat_history": history})


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: CONVERSATIONAL RAG")
    print("=" * 60)

    # The chat_history list grows with each turn
    chat_history = []

    conversation = [
        "What is LCEL?",
        "Can you give me a concrete code example of it?",  # follow-up
        "What about streaming — does it support that?",   # another follow-up
        "How is that different from the old LLMChain?",   # still in context
    ]

    for question in conversation:
        print(f"\nHuman: {question}")
        answer = chat(question, chat_history)
        print(f"AI:    {answer}")

        # Append this turn to history for the next call
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        print("─" * 60)
