import hashlib
import re
from collections.abc import Sequence
from itertools import cycle
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from etic_genai.documents.indexing.knowledge_base import KnowledgeBase
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.evaluation import JudgeVerdict


class FakeToolChatModel(GenericFakeChatModel):
    """Fake chat model that accepts `bind_tools`, so agents can be tested offline."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolChatModel":
        return self


class FakeJudgeModel(FakeToolChatModel):
    """Judge whose structured output returns the given scores in turn."""

    scores: list[float] = [1.0]
    calls: list[str] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        scores = cycle(self.scores)

        def judge(messages: Any) -> JudgeVerdict:
            self.calls.append(messages[0].content)
            return JudgeVerdict(score=next(scores), reasoning="fake verdict")

        return RunnableLambda(judge)


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic embeddings where texts sharing words are similar, so retrieval
    behaves meaningfully in tests without calling an API."""

    def __init__(self, size: int = 256) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self.size - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_llm_factory():
    def make(*replies: AIMessage | str) -> FakeToolChatModel:
        messages = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
        return FakeToolChatModel(messages=iter(messages))

    return make


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
def knowledge_base(tmp_path: Path, requirements_dir: Path) -> KnowledgeBase:
    return KnowledgeBase(
        store=QdrantStore(location=":memory:"),
        embeddings=BagOfWordsEmbeddings(),
        allowed_root=str(tmp_path),
    )
