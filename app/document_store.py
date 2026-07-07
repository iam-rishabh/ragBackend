import json
import os
import time
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, DATA_DIR, DOCUMENT_TTL_SECONDS, EMBEDDING_MODEL, HUGGINGFACEHUB_API_TOKEN
from app.hf_embeddings import HFRouterEmbeddings

META_PATH = os.path.join(DATA_DIR, "documents.json")

_loaders = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
}

embeddings = HFRouterEmbeddings(
    model=EMBEDDING_MODEL,
    api_token=HUGGINGFACEHUB_API_TOKEN,
)

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _load_meta() -> dict:
    if not os.path.exists(META_PATH):
        return {}

    with open(META_PATH, "r") as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError:
            return {}

    # Guard against the pre-session-scoping flat format ({document_id: entry}).
    # If the shape isn't {session_id: {document_id: entry}}, treat it as stale
    # and reset rather than crashing on malformed reads.
    for session_docs in meta.values():
        if not isinstance(session_docs, dict):
            return {}
        for entry in session_docs.values():
            if not isinstance(entry, dict) or "expires_at" not in entry:
                return {}

    return meta


def _save_meta(meta: dict):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _is_expired(entry: dict) -> bool:
    return time.time() >= entry.get("expires_at", 0)


def _purge_expired(meta: dict) -> bool:
    """Removes expired documents from the vector store and meta dict, in place.
    Returns True if anything was purged."""
    changed = False
    for session_id, docs in list(meta.items()):
        for document_id, entry in list(docs.items()):
            if _is_expired(entry):
                vectorstore._collection.delete(
                    where={"$and": [{"document_id": document_id}, {"session_id": session_id}]}
                )
                del docs[document_id]
                changed = True
        if not docs:
            del meta[session_id]
    return changed


def purge_expired_documents():
    """Called periodically by a background loop to sweep expired documents."""
    meta = _load_meta()
    if _purge_expired(meta):
        _save_meta(meta)


def list_documents(session_id: str) -> List[dict]:
    meta = _load_meta()
    if _purge_expired(meta):
        _save_meta(meta)

    session_docs = meta.get(session_id, {})
    now = time.time()
    return [
        {
            "id": d["id"],
            "name": d["name"],
            "size": d["size"],
            "chunks": d["chunks"],
            "expires_in": max(0, int(d["expires_at"] - now)),
        }
        for d in session_docs.values()
    ]


def document_count(session_id: str) -> int:
    meta = _load_meta()
    if _purge_expired(meta):
        _save_meta(meta)
    return len(meta.get(session_id, {}))


def ingest_file(file_path: str, filename: str, document_id: str, size: int, session_id: str) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    loader_cls = _loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")

    raw_docs = loader_cls(file_path).load()
    chunks = splitter.split_documents(raw_docs)

    expires_at = time.time() + DOCUMENT_TTL_SECONDS
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = filename
        chunk.metadata["session_id"] = session_id
        chunk.metadata["expires_at"] = expires_at

    if chunks:
        vectorstore.add_documents(chunks)

    meta = _load_meta()
    meta.setdefault(session_id, {})
    entry = {
        "id": document_id,
        "name": filename,
        "size": size,
        "chunks": len(chunks),
        "expires_at": expires_at,
    }
    meta[session_id][document_id] = entry
    _save_meta(meta)

    return {
        "id": document_id,
        "name": filename,
        "size": size,
        "chunks": len(chunks),
        "expires_in": DOCUMENT_TTL_SECONDS,
    }


def delete_document(document_id: str, session_id: str) -> bool:
    meta = _load_meta()
    session_docs = meta.get(session_id, {})
    if document_id not in session_docs:
        return False

    vectorstore._collection.delete(
        where={"$and": [{"document_id": document_id}, {"session_id": session_id}]}
    )
    del session_docs[document_id]
    _save_meta(meta)
    return True


def get_retriever(session_id: str, document_ids: Optional[List[str]] = None, k: int = 4):
    conditions = [{"session_id": session_id}, {"expires_at": {"$gt": time.time()}}]
    if document_ids:
        conditions.append({"document_id": {"$in": document_ids}})

    return vectorstore.as_retriever(search_kwargs={"k": k, "filter": {"$and": conditions}})