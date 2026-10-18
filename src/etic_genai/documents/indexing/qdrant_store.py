
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from fastembed import SparseEmbedding, SparseTextEmbedding
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from etic_genai.documents.models.DocumentChunk import DocumentChunk

QdrantMode = Literal["local", "docker"]


class QdrantStore:
    vector_size = 1536
    sparse_vector_name = "BM25"

    def __init__(
        self,
        path: str = "./data/qdrant",
        collection_name: str = "business_requirements",
        mode: QdrantMode = "local",
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if mode == "docker":
            if not url:
                raise ValueError(
                    "QDRANT_URL must be set when QDRANT_MODE=docker."
                )
            self.client = QdrantClient(url=url, api_key=api_key)
        elif mode == "local":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            raise ValueError(f"Unsupported Qdrant mode: {mode}")

        self.mode = mode
        self.collection_name = collection_name
        self._sparse_encoder: SparseTextEmbedding | None = None

        self._ensure_collection()

    @classmethod
    def from_settings(cls) -> "QdrantStore":
        from etic_genai.config import get_settings

        settings = get_settings()
        return cls(
            path=settings.qdrant_path or "./data/qdrant",
            collection_name=(
                settings.qdrant_collection_name or "business_requirements"
            ),
            mode=settings.qdrant_mode,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        collection_names = {
            collection.name for collection in collections
        }

        if self.collection_name not in collection_names:
            self._create_collection()
            return

        collection = self.client.get_collection(self.collection_name)
        sparse_vectors = dict(collection.config.params.sparse_vectors or {})
        if self.sparse_vector_name in sparse_vectors:
            return

        if self.mode == "local":
            point_count = self.client.count(self.collection_name, exact=True).count
            if point_count:
                existing_names = ", ".join(sorted(sparse_vectors)) or "none"
                raise RuntimeError(
                    f"Local Qdrant collection {self.collection_name!r} does not have "
                    f"the required sparse vector {self.sparse_vector_name!r} "
                    f"(configured: {existing_names}). Delete the collection after "
                    "removing its data, then restart and re-index your documents."
                )
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            return

        sparse_vectors[self.sparse_vector_name] = SparseVectorParams(
            modifier=Modifier.IDF
        )
        self.client.update_collection(
            collection_name=self.collection_name,
            sparse_vectors_config=sparse_vectors,
        )

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                self.sparse_vector_name: SparseVectorParams(
                    modifier=Modifier.IDF
                )
            },
        )

    def _get_sparse_encoder(self) -> SparseTextEmbedding:
        if self._sparse_encoder is None:
            self._sparse_encoder = SparseTextEmbedding(
                model_name="Qdrant/bm25",
                language="english",
            )
        return self._sparse_encoder

    @classmethod
    def _to_sparse_vector(
        cls,
        embedding: SparseEmbedding,
    ) -> dict[str, SparseVector]:
        indices = embedding.indices
        values = embedding.values

        if len(indices) != len(values):
            raise ValueError(
                "Sparse vector indices and values must have equal lengths."
            )

        if len(indices) == 0:
            return {}

        return {
            cls.sparse_vector_name: SparseVector(
                indices=[int(index) for index in indices],
                values=[float(value) for value in values],
            )
        }

    def upsert(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        owner_id: str | None = None,
        is_public: bool = False,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                "Each chunk must have exactly one dense vector."
            )

        if not chunks:
            return

        sparse_vectors = list(
            self._get_sparse_encoder().passage_embed(
                [chunk.content for chunk in chunks]
            )
        )

        if len(chunks) != len(sparse_vectors):
            raise RuntimeError(
                "Sparse encoder returned a different number of vectors "
                "than input chunks."
            )

        points = []

        for chunk, dense_vector, sparse_vector in zip(
            chunks, vectors, sparse_vectors, strict=True
        ):
            point_vectors = {
                "": dense_vector,
                **self._to_sparse_vector(sparse_vector),
            }

            points.append(
                PointStruct(
                    id=self._point_id(
                        f"{owner_id}:{chunk.chunk_id}"
                        if owner_id
                        else chunk.chunk_id
                    ),
                    vector=point_vectors,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "source": chunk.source,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "owner_id": owner_id,
                        "is_public": is_public,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    @traceable(name="qdrant-hybrid-retrieval")
    def search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 5,
        owner_id: str | None = None,
        public_only: bool = True,
    ) -> list:
        conditions = []

        if owner_id is not None:
            conditions.append(
                FieldCondition(
                    key="owner_id",
                    match=MatchValue(value=owner_id),
                )
            )

        if public_only:
            conditions.append(
                FieldCondition(
                    key="is_public",
                    match=MatchValue(value=True),
                )
            )

        query_filter = (
            Filter(should=conditions) if conditions else None
        )

        sparse_queries = list(
            self._get_sparse_encoder().query_embed(query_text)
        )

        if len(sparse_queries) != 1:
            raise RuntimeError(
                "Expected exactly one sparse query vector."
            )

        sparse_query = self._to_sparse_vector(sparse_queries[0])
        candidate_limit = max(limit * 4, 20)

        if sparse_query:
            return self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        limit=candidate_limit,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=sparse_query[self.sparse_vector_name],
                        using=self.sparse_vector_name,
                        limit=candidate_limit,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            ).points

        # Fall back to dense search if the query has no sparse terms.
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    @staticmethod
    def _point_id(chunk_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, chunk_id)
