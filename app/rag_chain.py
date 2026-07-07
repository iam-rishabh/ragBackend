import json
from typing import AsyncGenerator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.document_store import get_retriever
from app.schemas import ChatMessage

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using only the provided "
    "document context. If the context does not contain the answer, say you don't "
    "have enough information in the uploaded documents. Keep answers concise and "
    "reference the source file name in brackets, e.g. [handbook.pdf], when you use it."
)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2,
    streaming=True,
)


def _format_context(docs) -> str:
    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        parts.append(f"[{source}]\n{d.page_content}")
    return "\n\n".join(parts)


def _build_messages(question: str, history: List[ChatMessage], context: str):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append(HumanMessage(content=user_prompt))
    return messages


def _retrieve(question: str, document_ids: Optional[List[str]]):
    retriever = get_retriever(document_ids=document_ids)
    docs = retriever.invoke(question)
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return docs, sources


def answer(question: str, history: List[ChatMessage], document_ids: Optional[List[str]]) -> dict:
    docs, sources = _retrieve(question, document_ids)
    context = _format_context(docs)
    messages = _build_messages(question, history, context)
    result = llm.invoke(messages)
    return {"answer": result.content, "sources": sources}


async def stream_answer(
    question: str, history: List[ChatMessage], document_ids: Optional[List[str]]
) -> AsyncGenerator[str, None]:
    docs, sources = _retrieve(question, document_ids)
    context = _format_context(docs)
    messages = _build_messages(question, history, context)

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
