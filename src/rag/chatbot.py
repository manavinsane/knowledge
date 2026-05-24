from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.model.user import UserRole
from src.rag.access import metadata_filter_for_role

INDEX_NAME = "rag-bms"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


@lru_cache
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


@lru_cache
def get_llm():
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def create_retriever_for_role(role: UserRole | str):
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": 3,
            "filter": metadata_filter_for_role(role),
        }
    )


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def answer_question(question: str, role: UserRole | str) -> str:
    retriever = create_retriever_for_role(role)
    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain.invoke(question)
