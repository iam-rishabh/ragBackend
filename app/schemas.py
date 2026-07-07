from typing import List, Optional
from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    name: str
    size: int
    chunks: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    document_ids: Optional[List[str]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
