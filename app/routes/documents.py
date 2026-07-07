import os
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import DATA_DIR, MAX_DOCUMENTS
from app.document_store import delete_document, document_count, ingest_file, list_documents
from app.schemas import DocumentOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXT = {".pdf", ".doc", ".docx", ".txt"}


@router.get("", response_model=List[DocumentOut])
def get_documents():
    return list_documents()


@router.post("/upload", response_model=List[DocumentOut])
async def upload_documents(files: List[UploadFile] = File(...)):
    existing = document_count()
    if existing + len(files) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {MAX_DOCUMENTS} documents allowed, {existing} already uploaded.",
        )

    results = []
    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename}")

        document_id = str(uuid.uuid4())
        temp_path = os.path.join(DATA_DIR, f"{document_id}{ext}")
        content = await upload.read()

        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            entry = ingest_file(temp_path, upload.filename, document_id, len(content))
        finally:
            os.remove(temp_path)

        results.append(entry)

    return results


@router.delete("/{document_id}")
def remove_document(document_id: str):
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"id": document_id, "deleted": True}
