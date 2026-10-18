from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastembed import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    PointStruct,
    SparseVector,
    VectorParams,
)

from etic_genai.documents.indexing import qdrant_store as qdrant_store_module
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.documents.models.DocumentChunk import DocumentChunk


class FakeBM25Encoder:
    def passage_embed(self, texts):
        for text in texts:
            indices = [11, 13] if "AUD-104" in text else [17]
            yield SparseEmbedding(
                indices=np.asarray(indices, dtype=np.int64),
                values=np.asarray([1.7, 0.8][: len(indices)], dtype=np.float32),
            )

    def query_embed(self, text):
        indices = [11, 13] if "AUD-104" in text else []
        yield SparseEmbedding(
            indices=np.asarray(indices, dtype=np.int64),
            values=np.asarray([1.0, 1.0][: len(indices)], dtype=np.float32),
        )


class FakeQdrantClient:
    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[])

    def create_collection(self, **_kwargs: object) -> None:
        pass


def test_qdrant_store_connects_to_server_when_url_is_set(monkeypatch):
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> FakeQdrantClient:
        captured.update(kwargs)
        return FakeQdrantClient()

    monkeypatch.setattr(qdrant_store_module, "QdrantClient", fake_client)

    QdrantStore(mode="docker", url="http://localhost:6333", api_key="test-key")

    assert captured == {"url": "http://localhost:6333", "api_key": "test-key"}


def test_qdrant_store_keeps_embedded_storage_without_url(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> FakeQdrantClient:
        captured.update(kwargs)
        return FakeQdrantClient()

    monkeypatch.setattr(qdrant_store_module, "QdrantClient", fake_client)
    local_path = tmp_path / "qdrant"

    QdrantStore(mode="local", path=str(local_path))

    assert captured == {"path": str(local_path)}


def test_docker_mode_requires_server_url(monkeypatch):
    monkeypatch.setattr(
        qdrant_store_module,
        "QdrantClient",
        lambda **_kwargs: FakeQdrantClient(),
    )

    with pytest.raises(ValueError, match="QDRANT_URL"):
        QdrantStore(mode="docker")


def test_search_can_filter_for_public_documents():
    captured: dict[str, object] = {}

    class SearchClient:
        def query_points(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    store = QdrantStore.__new__(QdrantStore)
    store.client = SearchClient()
    store.collection_name = "documents"
    store._sparse_encoder = FakeBM25Encoder()

    store.search([0.1] * 1536, query_text="AUD-104", owner_id="user-1")
    owner_filter = captured["query_filter"]
    assert [condition.key for condition in owner_filter.should] == ["owner_id", "is_public"]
    assert owner_filter.should[0].match.value == "user-1"
    assert owner_filter.should[1].match.value is True
    assert isinstance(captured["query"], FusionQuery)
    assert captured["query"].fusion == Fusion.RRF
    assert len(captured["prefetch"]) == 2
    assert captured["prefetch"][0].filter == owner_filter
    assert captured["prefetch"][1].filter == owner_filter
    assert captured["prefetch"][1].using == "BM25"
    assert isinstance(captured["prefetch"][1].query, SparseVector)

    store.search([0.1] * 1536, query_text="AUD-104", public_only=True)
    public_filter = captured["query_filter"]
    assert [condition.key for condition in public_filter.should] == ["is_public"]
    assert public_filter.should[0].match.value is True


def test_upsert_stores_dense_and_sparse_vectors():
    captured: dict[str, object] = {}

    class UpsertClient:
        def upsert(self, **kwargs: object) -> None:
            captured.update(kwargs)

    store = QdrantStore.__new__(QdrantStore)
    store.client = UpsertClient()
    store.collection_name = "documents"
    store._sparse_encoder = FakeBM25Encoder()
    chunk = DocumentChunk(
        document_id="doc-1",
        document_title="Requirements",
        source="requirements.txt",
        content="AUD-104 requires audit logging.",
    )

    store.upsert([chunk], [[0.1] * 1536], owner_id="user-1")

    point = captured["points"][0]
    assert point.vector[""] == [0.1] * 1536
    assert isinstance(point.vector["BM25"], SparseVector)
    assert point.vector["BM25"].indices == [11, 13]
    assert point.vector["BM25"].values == pytest.approx([1.7, 0.8])


def test_bm25_embedding_is_converted_to_qdrant_sparse_vector():
    vector = QdrantStore._to_sparse_vector(
        SparseEmbedding(
            indices=np.asarray([11, 13], dtype=np.int64),
            values=np.asarray([1.7, 0.8], dtype=np.float32),
        )
    )

    assert vector["BM25"] == SparseVector(
        indices=[11, 13],
        values=[float(np.float32(1.7)), float(np.float32(0.8))],
    )
    assert (
        QdrantStore._to_sparse_vector(
            SparseEmbedding(indices=np.asarray([], dtype=np.int64), values=np.asarray([]))
        )
        == {}
    )


def test_bm25_embedding_rejects_mismatched_indices_and_values():
    with pytest.raises(ValueError, match="indices and values"):
        QdrantStore._to_sparse_vector(
            SparseEmbedding(
                indices=np.asarray([11], dtype=np.int64),
                values=np.asarray([], dtype=np.float32),
            )
        )


def test_search_uses_dense_only_for_queries_without_tokens():
    captured: dict[str, object] = {}

    class SearchClient:
        def query_points(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(points=[])

    store = QdrantStore.__new__(QdrantStore)
    store.client = SearchClient()
    store.collection_name = "documents"
    store._sparse_encoder = FakeBM25Encoder()

    store.search([0.1] * 1536, query_text="!!!")

    assert captured["query"] == [0.1] * 1536
    assert "prefetch" not in captured


def test_empty_local_dense_collection_is_recreated_with_sparse_schema():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="legacy",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    store = QdrantStore.__new__(QdrantStore)
    store.client = client
    store.collection_name = "legacy"
    store.mode = "local"

    store._ensure_collection()
    collection = client.get_collection("legacy")
    assert collection.config.params.sparse_vectors["BM25"].modifier is not None
    assert client.count("legacy", exact=True).count == 0
    client.close()


def test_nonempty_local_dense_collection_requires_explicit_deletion():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="legacy",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name="legacy",
        points=[PointStruct(id=1, vector=[1.0] + [0.0] * 1535)],
    )
    store = QdrantStore.__new__(QdrantStore)
    store.client = client
    store.collection_name = "legacy"
    store.mode = "local"

    with pytest.raises(RuntimeError, match="Delete the collection"):
        store._ensure_collection()

    assert client.count("legacy", exact=True).count == 1
    client.close()


def test_nonempty_local_collection_with_old_sparse_name_requires_explicit_deletion():
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="legacy",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        sparse_vectors_config={"lexical": qdrant_store_module.SparseVectorParams()},
    )
    client.upsert(
        collection_name="legacy",
        points=[PointStruct(id=1, vector={"": [1.0] + [0.0] * 1535})],
    )
    store = QdrantStore.__new__(QdrantStore)
    store.client = client
    store.collection_name = "legacy"
    store.mode = "local"

    with pytest.raises(RuntimeError, match="'BM25'.*configured: lexical"):
        store._ensure_collection()

    assert client.count("legacy", exact=True).count == 1
    client.close()


def test_local_hybrid_search_can_rank_an_exact_term_over_dense_only_match():
    client = QdrantClient(":memory:")
    store = QdrantStore.__new__(QdrantStore)
    store.client = client
    store.collection_name = "hybrid"
    store._sparse_encoder = FakeBM25Encoder()
    store._create_collection()
    store.upsert(
        [
            DocumentChunk(
                document_id="generic",
                document_title="Generic",
                source="generic.txt",
                content="General process guidance.",
            ),
            DocumentChunk(
                document_id="audit",
                document_title="Audit",
                source="audit.txt",
                content="AUD-104 requires audit logs.",
            ),
        ],
        [
            [1.0] + [0.0] * 1535,
            [0.0, 1.0] + [0.0] * 1534,
        ],
        owner_id="user-1",
    )

    results = store.search(
        [1.0] + [0.0] * 1535,
        query_text="AUD-104",
        owner_id="user-1",
    )

    assert results[0].payload["document_id"] == "audit"
    client.close()
