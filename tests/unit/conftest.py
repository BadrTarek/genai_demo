"""Fixtures for the shared knowledge base, guarded RAG agent and MCP server tests.

Everything runs offline: bag-of-words dense embeddings, a word-hashing BM25 encoder and
a real local Qdrant store in a temp folder, so hybrid search (RRF) is exercised for real.
"""

import hashlib
import re
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastembed import SparseEmbedding
from langchain_core.runnables import RunnableLambda

from etic_genai.documents.indexing.knowledge_base import SharedKnowledgeBase
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.guardrails.output import GroundednessVerdict
from tests.conftest import FakeToolChatModel


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]


def _bucket(word: str, size: int) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % size


class BagOfWordsEmbeddings:
    """Dense fake: texts sharing words are similar. 1536-d to match `QdrantStore`."""

    size = QdrantStore.vector_size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in _words(text):
            vector[_bucket(word, self.size)] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self.size - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class WordHashBM25:
    """Sparse fake with the fastembed `passage_embed` / `query_embed` interface."""

    def _embed(self, text: str) -> SparseEmbedding:
        counts = Counter(_bucket(word, 100_000) for word in _words(text))
        return SparseEmbedding(
            indices=np.asarray(list(counts), dtype=np.int64),
            values=np.asarray(list(counts.values()), dtype=np.float32),
        )

    def passage_embed(self, texts: list[str]):
        for text in texts:
            yield self._embed(text)

    def query_embed(self, text: str):
        yield self._embed(text)


class FakeJudgeModel(FakeToolChatModel):
    """Judge whose structured output returns the given scores in turn."""

    scores: list[float] = [1.0]
    calls: list[str] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        scores = cycle(self.scores)

        def judge(messages: Any) -> GroundednessVerdict:
            self.calls.append(messages[0].content)
            return GroundednessVerdict(score=next(scores), reasoning="fake verdict")

        return RunnableLambda(judge)


@pytest.fixture
def fake_judge_factory():
    def make(*scores: float) -> FakeJudgeModel:
        return FakeJudgeModel(messages=iter([]), scores=list(scores) or [1.0], calls=[])

    return make


@pytest.fixture
def requirements_dir(tmp_path: Path) -> Path:
    root = tmp_path / "requirements"
    root.mkdir()
    (root / "authentication.md").write_text(
        "# Authentication\n\n## Passwords\n\n"
        "- Passwords must contain at least 12 characters.\n"
        "- The system must lock an account after five failed login attempts.\n",
        encoding="utf-8",
    )
    (root / "invoice_approval.txt").write_text(
        "Invoices from 1,000 EUR up to 10,000 EUR require approval from a finance manager.\n"
        "The invoice creator cannot approve their own invoice.\n",
        encoding="utf-8",
    )
    (root / "user_roles.csv").write_text(
        "role,permission,description\nfinance_manager,invoice.reopen,Reopen approved invoices\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def qdrant_store(tmp_path: Path) -> QdrantStore:
    store = QdrantStore(path=str(tmp_path / "qdrant"), mode="local")
    store._sparse_encoder = WordHashBM25()
    yield store
    store.client.close()


@pytest.fixture
def knowledge_base(qdrant_store: QdrantStore, tmp_path: Path) -> SharedKnowledgeBase:
    return SharedKnowledgeBase(
        embedding_model=BagOfWordsEmbeddings(),
        qdrant_store=qdrant_store,
        allowed_root=str(tmp_path),
    )


@pytest.fixture
def indexed_kb(knowledge_base: SharedKnowledgeBase, requirements_dir: Path) -> SharedKnowledgeBase:
    knowledge_base.index_directory(str(requirements_dir))
    return knowledge_base
