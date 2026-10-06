from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from etic_genai.documents.models.DocumentChunk import DocumentChunk


class QdrantStore:
    vector_size = 1536

    def __init__(
        self,
        path: str = "./data/qdrant",
        collection_name: str = "business_requirements",
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=path)
        self.collection_name = collection_name

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = {collection.name for collection in collections}

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")

        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list:
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points

    @staticmethod
    def _point_id(chunk_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, chunk_id)