"""Qdrant persistence for document chunks.

Every point carries the indexed `root` and `document_id` in its payload, so a document
can be replaced or removed as a unit and several roots can share one collection.
"""

import atexit
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from etic_genai.core.exceptions import ConfigurationError
from etic_genai.core.logging import get_logger
from etic_genai.documents.models.DocumentChunk import DocumentChunk

logger = get_logger(__name__)


class QdrantStore:
    def __init__(
        self,
        collection_name: str = "business_requirements",
        path: str | None = "./data/qdrant",
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
    ) -> None:
        """Connect to a Qdrant server (`url`), an in-process store (`location=":memory:"`)
        or a local on-disk store (`path`). A local store can only be opened by one process
        at a time; run a Qdrant server when the MCP server and the agent run together."""
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        elif location:
            self.client = QdrantClient(location=location)
        self.local_path = None if url or location else Path(path or "./data/qdrant")
        if self.local_path is not None:
            self.local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.local_path))
            # Release the file lock before interpreter teardown (avoids noisy __del__ errors).
            atexit.register(self.client.close)
        self.collection_name = collection_name

    # -- collection -------------------------------------------------------------------

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return
        existing_size = self.client.get_collection(self.collection_name).config.params.vectors.size
        if existing_size != vector_size:
            fix = (
                f"delete the folder {self.local_path} and re-index"
                if self.local_path is not None
                else "re-index with --reset"
            )
            raise ConfigurationError(
                f"Collection '{self.collection_name}' stores {existing_size}-d vectors but the "
                f"embedding model returns {vector_size}-d vectors: {fix}."
            )

    def reset(self) -> None:
        """Remove every point. In local mode the collection is emptied rather than dropped:
        on Windows the local client keeps the collection's SQLite file open, so a dropped
        collection stays on disk and is reloaded, old points included, when re-created."""
        if not self.exists():
            return
        if self.local_path is None:
            self.client.delete_collection(self.collection_name)
        else:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=Filter()),
            )

    # -- writes -----------------------------------------------------------------------

    def replace_document(
        self, root: str, document_id: str, chunks: list[DocumentChunk], vectors: list[list[float]]
    ) -> None:
        """Atomically (from the caller's view) swap all chunks of one document."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")
        if vectors:
            self._ensure_collection(len(vectors[0]))
        self.delete_documents(root, [document_id])
        if not chunks:
            return
        points = [
            PointStruct(
                id=self._point_id(root, chunk.chunk_id),
                vector=vector,
                payload={
                    "root": root,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "document_hash": chunk.document_hash,
                    "file_type": chunk.file_type,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_documents(self, root: str, document_ids: Iterable[str]) -> None:
        document_ids = list(document_ids)
        if not document_ids or not self.exists():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="root", match=MatchValue(value=root)),
                        FieldCondition(key="document_id", match=MatchAny(any=document_ids)),
                    ]
                )
            ),
        )

    # -- reads ------------------------------------------------------------------------

    def _scroll_payloads(self, scroll_filter: Filter | None = None) -> Iterable[dict[str, Any]]:
        if not self.exists():
            return
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=["root", "document_id", "document_title", "document_hash", "source"],
                with_vectors=False,
            )
            for point in points:
                yield point.payload or {}
            if offset is None:
                break

    def document_hashes(self, root: str) -> dict[str, str]:
        """`{document_id: content_hash}` for every document indexed under `root`."""
        root_filter = Filter(must=[FieldCondition(key="root", match=MatchValue(value=root))])
        return {
            payload["document_id"]: payload.get("document_hash", "")
            for payload in self._scroll_payloads(root_filter)
        }

    def list_documents(self) -> list[dict[str, Any]]:
        documents: dict[tuple[str, str], dict[str, Any]] = {}
        for payload in self._scroll_payloads():
            key = (payload.get("root", ""), payload.get("document_id", ""))
            entry = documents.setdefault(
                key,
                {
                    "root": key[0],
                    "document_id": key[1],
                    "title": payload.get("document_title"),
                    "source": payload.get("source"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        return sorted(documents.values(), key=lambda doc: (doc["root"], doc["document_id"]))

    def search(self, query_vector: list[float], limit: int = 5) -> list:
        if not self.exists():
            return []
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

    @staticmethod
    def _point_id(root: str, chunk_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"{root}|{chunk_id}")
