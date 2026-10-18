"""Document upload and indexing endpoints."""

import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from etic_genai.api.dependencies import IngestionServices, get_current_user, get_ingestion_services
from etic_genai.api.schemas import ChunkResponse, UploadResponse
from etic_genai.auth.store import AuthUser
from etic_genai.documents.models.DocumentChunk import DocumentChunk

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    description=(
        "Accepts PDF, Markdown, text, or CSV files up to 10 MiB. "
        "The file is chunked, embedded, and stored in Qdrant."
    ),
    responses={
        401: {"description": "A valid bearer token is required."},
        413: {"description": "The upload exceeds the 10 MiB limit."},
        415: {"description": "The file extension is not supported."},
    },
)
async def upload_document(
    file: Annotated[
        UploadFile,
        File(description="PDF, Markdown, text, or CSV file; maximum 10 MiB."),
    ],
    services: Annotated[IngestionServices, Depends(get_ingestion_services)],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    visibility: Annotated[
        Literal["private", "public"],
        Form(description="Public uploads can be used as fallback context for other users."),
    ] = "private",
) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="A document filename is required.")

    try:
        loader = services.loader_factory.get_loader(Path(filename))
    except ValueError as exc:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Supported types are PDF, Markdown, text, and CSV.",
        ) from exc

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            document_root = Path(temp_dir)
            file_path = document_root / filename
            total_bytes = 0
            with file_path.open("wb") as destination:
                while content := await file.read(1024 * 1024):
                    total_bytes += len(content)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="The document exceeds the 10 MiB upload limit.",
                        )
                    destination.write(content)

            document = loader.load(file_path, document_root)
    finally:
        await file.close()

    chunks: list[DocumentChunk] = services.chunker.split(document)
    vectors = services.embedding_model.embed_documents([chunk.content for chunk in chunks])
    services.qdrant_store.upsert(
        chunks,
        vectors,
        owner_id=current_user.user_id,
        is_public=visibility == "public",
    )

    return UploadResponse(
        document_id=document.document_id,
        document_title=document.title,
        chunk_count=len(chunks),
        chunks=[
            ChunkResponse(
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                content=chunk.content,
            )
            for chunk in chunks
        ],
    )
