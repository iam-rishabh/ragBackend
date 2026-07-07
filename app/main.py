import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS, PURGE_INTERVAL_SECONDS
from app.document_store import purge_expired_documents
from app.routes import chat, documents


async def _purge_loop():
    while True:
        try:
            purge_expired_documents()
        except Exception as exc:
            print(f"[purge_loop] error: {exc}")
        await asyncio.sleep(PURGE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_purge_loop())
    yield
    task.cancel()


app = FastAPI(title="Archive RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}