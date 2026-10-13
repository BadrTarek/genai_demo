"""Reusable document ingestion and vector search service."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.documents.loaders.factory import DocumentLoaderFactory
from etic_genai.documents.processing.chunker import DocumentChunker
from etic_genai.embeddings.model import EmbeddingModel


class KnowledgeBase:
    """Load documents from a user-approved path and search their chunks."""

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant",
        qdrant_url: str | None = None,
        collection_name: str = "business_requirements",
        allowed_root: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        load_dotenv()
        self.allowed_root = Path(allowed_root).resolve() if allowed_root else None
        self.factory = DocumentLoaderFactory()
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.embedding_model = EmbeddingModel(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.store = QdrantStore(
            path=qdrant_path,
            url=qdrant_url,
            collection_name=collection_name,
            vector_size=None,
        )

    def _validate_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"Knowledge-base path is not a directory: {path}")
        if self.allowed_root and self.allowed_root not in (resolved, *resolved.parents):
            raise ValueError(
                f"Knowledge-base path is outside the allowed root: {self.allowed_root}"
            )
        return resolved

    def index_path(self, path: str, output_path: str | None = None) -> dict[str, Any]:
        root = self._validate_path(path)
        indexed: dict[str, list[dict[str, Any]]] = {}
        skipped: list[str] = []

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                loader = self.factory.get_loader(file_path)
            except ValueError:
                skipped.append(str(file_path))
                continue

            document = loader.load(file_path, root)
            chunks = self.chunker.split(document)
            vectors = self.embedding_model.embed_documents([chunk.content for chunk in chunks])
            self.store.upsert(chunks, vectors)
            indexed[document.document_id] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                }
                for chunk in chunks
            ]

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(indexed, indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "path": str(root),
            "documents_indexed": len(indexed),
            "chunks_indexed": sum(len(chunks) for chunks in indexed.values()),
            "files_skipped": len(skipped),
            "skipped": skipped,
        }

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        query_vector = self.embedding_model.embed_query(query)
        return [
            {"score": point.score, **(point.payload or {})}
            for point in self.store.search(query_vector, limit=max(1, min(limit, 50)))
        ]