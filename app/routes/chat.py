import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.deps import get_session_id
from app.guardrails import looks_like_injection
from app.rag_chain import answer, stream_answer
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])

REFUSAL = "I can't share internal instructions or system prompts. Ask me something about your uploaded documents instead."


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, session_id: str = Depends(get_session_id)):
    if looks_like_injection(req.message):
        return {"answer": REFUSAL, "sources": []}

    result = answer(req.message, req.history, req.document_ids, session_id)
    return result


@router.post("/stream")
async def chat_stream(req: ChatRequest, session_id: str = Depends(get_session_id)):
    if looks_like_injection(req.message):
        async def refusal_stream():
            yield f"data: {json.dumps({'type': 'token', 'content': REFUSAL})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(refusal_stream(), media_type="text/event-stream")

    generator = stream_answer(req.message, req.history, req.document_ids, session_id)
    return StreamingResponse(generator, media_type="text/event-stream")