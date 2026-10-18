"""Document ingestion and vector search service shared by the CLI, MCP server and agent."""

import json
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from etic_genai.config import Settings, get_settings
from etic_genai.core.logging import get_logger
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.documents.loaders.factory import DocumentLoaderFactory
from etic_genai.documents.processing.chunker import DocumentChunker

logger = get_logger(__name__)

# Directories that never contain knowledge-base documents.
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}


class KnowledgeBase:
    """Index a directory into the vector store and search it.

    Indexing is incremental: a document is re-embedded only when its content hash
    changes, and documents deleted from disk are removed from the store.
    """

    def __init__(
        self,
        store: QdrantStore,
        embeddings: Embeddings,
        chunker: DocumentChunker | None = None,
        loader_factory: DocumentLoaderFactory | None = None,
        allowed_root: str | None = None,
        max_files: int = 1000,
        max_file_size_mb: float = 20.0,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chunker = chunker or DocumentChunker()
        self.loader_factory = loader_factory or DocumentLoaderFactory()
        self.allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self.max_files = max_files
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "KnowledgeBase":
        from etic_genai.embeddings import get_embeddings

        settings = settings or get_settings()
        return cls(
            store=QdrantStore(
                collection_name=settings.qdrant_collection_name,
                path=settings.qdrant_path,
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            ),
            embeddings=get_embeddings(settings),
            chunker=DocumentChunker(settings.chunk_size, settings.chunk_overlap),
            allowed_root=settings.knowledge_base_root or settings.knowledge_base_path,
            max_files=settings.max_files_per_index,
            max_file_size_mb=settings.max_file_size_mb,
        )

    def _validate_path(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Knowledge-base path is not a directory: {path}")
        if self.allowed_root and self.allowed_root not in (resolved, *resolved.parents):
            raise ValueError(
                f"Knowledge-base path is outside the allowed root: {self.allowed_root}"
            )
        return resolved

    def _candidate_files(self, root: Path) -> list[Path]:
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not IGNORED_DIRS.intersection(path.relative_to(root).parts)
            and self.loader_factory.supports(path)
        ]
        if len(files) > self.max_files:
            raise ValueError(
                f"{root} contains {len(files)} supported files (limit {self.max_files}). "
                "Point the knowledge base at a narrower directory."
            )
        return files

    def index_path(
        self, path: str, output_path: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        root = self._validate_path(path)
        root_key = root.as_posix()
        existing = self.store.document_hashes(root_key)
        seen: set[str] = set()
        indexed: dict[str, list[dict[str, Any]]] = {}
        unchanged: list[str] = []
        failed: list[dict[str, str]] = []

        for file_path in self._candidate_files(root):
            document_id = file_path.relative_to(root).as_posix()
            # Mark as seen before loading so a temporarily unreadable file keeps its old chunks.
            seen.add(document_id)
            try:
                if file_path.stat().st_size > self.max_file_size_bytes:
                    raise ValueError("file exceeds the size limit")
                document = self.loader_factory.get_loader(file_path).load(file_path, root)
                if not force and existing.get(document_id) == document.content_hash:
                    unchanged.append(document_id)
                    continue
                chunks = self.chunker.split(document)
                vectors = (
                    self.embeddings.embed_documents([chunk.embedding_text for chunk in chunks])
                    if chunks
                    else []
                )
                self.store.replace_document(root_key, document_id, chunks, vectors)
            except Exception as exc:
                logger.warning("Failed to index %s: %s", file_path, exc)
                failed.append({"document_id": document_id, "error": str(exc)})
                continue

            indexed[document_id] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_title": chunk.document_title,
                    "section": chunk.section,
                    "content": chunk.content,
                }
                for chunk in chunks
            ]

        removed = sorted(set(existing) - seen)
        self.store.delete_documents(root_key, removed)

        if output_path and indexed:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(indexed, indent=2, ensure_ascii=False), encoding="utf-8")

        report = {
            "path": root_key,
            "documents_indexed": len(indexed),
            "chunks_indexed": sum(len(chunks) for chunks in indexed.values()),
            "documents_unchanged": len(unchanged),
            "documents_removed": removed,
            "failed": failed,
        }
        logger.info("Indexed %s: %s", root_key, {k: v for k, v in report.items() if k != "path"})
        return report

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        query_vector = self.embeddings.embed_query(query)
        return [
            {"score": round(point.score, 4), **(point.payload or {})}
            for point in self.store.search(query_vector, limit=max(1, min(top_k, 50)))
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()

    def reset(self) -> None:
        self.store.reset()
