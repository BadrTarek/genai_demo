"""Shared (public) knowledge base on top of the hybrid `QdrantStore`.

Used by the MCP server and the guarded RAG agent. Documents indexed here are stored
as public (`owner_id=None`, `is_public=True`), so they are also the public fallback
context for users of the web API.
"""

import math
from pathlib import Path
from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    IsNullCondition,
    MatchValue,
    PayloadField,
)

from etic_genai.core.logging import get_logger
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.documents.loaders.factory import DocumentLoaderFactory
from etic_genai.documents.processing.chunker import DocumentChunker
from etic_genai.embeddings.model import EmbeddingModel

logger = get_logger(__name__)

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def _shared_document_filter(document_id: str | None = None) -> Filter:
    must: list[Any] = [
        FieldCondition(key="is_public", match=MatchValue(value=True)),
        IsNullCondition(is_null=PayloadField(key="owner_id")),
    ]
    if document_id is not None:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    return Filter(must=must)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SharedKnowledgeBase:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        qdrant_store: QdrantStore,
        loader_factory: DocumentLoaderFactory | None = None,
        chunker: DocumentChunker | None = None,
        allowed_root: str | None = None,
        max_files: int = 1000,
    ) -> None:
        self.embedding_model = embedding_model
        self.store = qdrant_store
        self.loader_factory = loader_factory or DocumentLoaderFactory()
        self.chunker = chunker or DocumentChunker()
        self.allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self.max_files = max_files

    @classmethod
    def from_settings(cls) -> "SharedKnowledgeBase":
        # Reuse the API's service factory so both use the same embeddings and store.
        from etic_genai.api.dependencies import get_ingestion_services
        from etic_genai.config import get_settings

        settings = get_settings()
        services = get_ingestion_services()
        return cls(
            embedding_model=services.embedding_model,
            qdrant_store=services.qdrant_store,
            loader_factory=services.loader_factory,
            chunker=services.chunker,
            allowed_root=settings.knowledge_base_root or settings.requirements_dir,
        )

    # -- indexing -----------------------------------------------------------------------

    def _validate_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Knowledge-base path is not a directory: {path}")
        if self.allowed_root and self.allowed_root not in (resolved, *resolved.parents):
            raise ValueError(
                f"Knowledge-base path is outside the allowed root: {self.allowed_root}"
            )
        return resolved

    def index_directory(self, path: str) -> dict[str, Any]:
        """Index every supported file under `path` as shared documents.

        Re-indexing a document replaces all of its previous chunks, so edited files
        leave no stale chunks behind. One unreadable file does not stop the run.
        """
        root = self._validate_path(path)
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and not IGNORED_DIRS.intersection(p.relative_to(root).parts)
        ]
        if len(files) > self.max_files:
            raise ValueError(f"{root} contains {len(files)} files (limit {self.max_files}).")

        indexed: dict[str, int] = {}
        skipped: list[str] = []
        failed: list[dict[str, str]] = []
        for file_path in files:
            document_id = file_path.relative_to(root).as_posix()
            try:
                loader = self.loader_factory.get_loader(file_path)
            except ValueError:
                skipped.append(document_id)
                continue
            try:
                document = loader.load(file_path, root)
                chunks = self.chunker.split(document)
                vectors = self.embedding_model.embed_documents([c.content for c in chunks])
                self.store.client.delete(
                    collection_name=self.store.collection_name,
                    points_selector=FilterSelector(filter=_shared_document_filter(document_id)),
                )
                self.store.upsert(chunks, vectors, owner_id=None, is_public=True)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", file_path, exc)
                failed.append({"document_id": document_id, "error": str(exc)})
                continue
            indexed[document_id] = len(chunks)

        return {
            "path": root.as_posix(),
            "documents_indexed": len(indexed),
            "chunks_indexed": sum(indexed.values()),
            "skipped": skipped,
            "failed": failed,
        }

    # -- reads --------------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Hybrid (dense + BM25, RRF-fused) search over shared documents.

        Each hit has the fused rank `score` and a `relevance`: the dense cosine similarity
        between the query and the passage. RRF scores only rank results, so `relevance`
        is what callers should threshold on.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        query_vector = self.embedding_model.embed_query(query)
        points = self.store.search(
            query_vector, query_text=query, limit=max(1, min(top_k, 50)), public_only=True
        )
        if not points:
            return []
        # Requesting only the unnamed dense vector returns it as a plain list; with
        # several vectors it is the "" entry of a dict.
        dense = {
            record.id: (
                record.vector.get("", []) if isinstance(record.vector, dict) else record.vector
            )
            or []
            for record in self.store.client.retrieve(
                collection_name=self.store.collection_name,
                ids=[point.id for point in points],
                with_vectors=[""],
            )
        }
        return [
            {
                "score": round(float(point.score), 4),
                "relevance": round(_cosine(query_vector, dense.get(point.id, [])), 4),
                **{
                    key: (point.payload or {}).get(key)
                    for key in ("document_id", "document_title", "source", "section", "content")
                },
            }
            for point in points
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self.store.client.scroll(
                collection_name=self.store.collection_name,
                scroll_filter=_shared_document_filter(),
                limit=256,
                offset=offset,
                with_payload=["document_id", "document_title", "source"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                entry = documents.setdefault(
                    payload.get("document_id", ""),
                    {
                        "document_id": payload.get("document_id"),
                        "title": payload.get("document_title"),
                        "source": payload.get("source"),
                        "chunks": 0,
                    },
                )
                entry["chunks"] += 1
            if offset is None:
                break
        return sorted(documents.values(), key=lambda doc: doc["document_id"] or "")
