import json
import os
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, DATA_DIR, EMBEDDING_MODEL

META_PATH = os.path.join(DATA_DIR, "documents.json")

_loaders = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
}

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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
        return json.load(f)


def _save_meta(meta: dict):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def list_documents() -> List[dict]:
    return list(_load_meta().values())


def document_count() -> int:
    return len(_load_meta())


def ingest_file(file_path: str, filename: str, document_id: str, size: int) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    loader_cls = _loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")

    raw_docs = loader_cls(file_path).load()
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = filename

    if chunks:
        vectorstore.add_documents(chunks)

    meta = _load_meta()
    entry = {"id": document_id, "name": filename, "size": size, "chunks": len(chunks)}
    meta[document_id] = entry
    _save_meta(meta)
    return entry


def delete_document(document_id: str) -> bool:
    meta = _load_meta()
    if document_id not in meta:
        return False

    vectorstore._collection.delete(where={"document_id": document_id})
    del meta[document_id]
    _save_meta(meta)
    return True


def get_retriever(document_ids: Optional[List[str]] = None, k: int = 4):
    search_kwargs = {"k": k}
    if document_ids:
        search_kwargs["filter"] = {"document_id": {"$in": document_ids}}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
