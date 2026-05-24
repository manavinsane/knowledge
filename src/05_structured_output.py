"""
STEP 5: Structured Output with Pydantic
=========================================
Covers: with_structured_output, Pydantic BaseModel, Field,
        getting typed objects back instead of raw strings

with_structured_output uses tool-calling internally — the LLM is
forced to return JSON that validates against your Pydantic schema.
This is the most reliable way to get structured data from an LLM.
"""
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

INDEX_NAME = "rag-bms"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Define the output schema ──────────────────────────────────────
# Field(description=...) tells the LLM what each field should contain.
# Good field descriptions → better LLM output.
class RAGAnswer(BaseModel):
    """Structured answer produced by the RAG chain."""

    answer: str = Field(description="Direct answer to the question in 2-4 sentences")
    key_concepts: list[str] = Field(description="2-4 key concepts mentioned in the answer")
    confidence: float = Field(description="Confidence from 0.0 (not sure) to 1.0 (certain)")
    follow_up_questions: list[str] = Field(
        description="2 related questions the user might want to explore next"
    )


# ── Structured LLM ───────────────────────────────────────────────
# with_structured_output wraps the LLM so it ALWAYS returns a RAGAnswer object.
# Under the hood it uses tool calling — no fragile text parsing.
structured_llm = llm.with_structured_output(RAGAnswer)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using ONLY the context below.
If you cannot find the answer, set confidence to 0.1 and say so.

Context:
{context}"""),
    ("human", "{question}"),
])

structured_rag = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["question"]))
    )
    | prompt
    | structured_llm   # returns RAGAnswer, not str
)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: STRUCTURED OUTPUT")
    print("=" * 60)

    questions = [
        "What is LCEL and what are its benefits?",
        "What retriever types are available in LangChain?",
        "How do agents work in LangChain?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result: RAGAnswer = structured_rag.invoke({"question": q})

        print(f"Answer      : {result.answer}")
        print(f"Confidence  : {result.confidence:.0%}")
        print(f"Key Concepts: {', '.join(result.key_concepts)}")
        print("Follow-ups  :")
        for fq in result.follow_up_questions:
            print(f"  • {fq}")
        print("─" * 60)
