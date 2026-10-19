"""MCP server exposing the shared knowledge base to MCP clients such as GitHub Copilot.

    python -m etic_genai.mcp_server

Tools:
- `index_documents`    index a directory (under the allowed root) as shared documents
- `search_documents`   hybrid (dense + BM25) search; the client's own model writes the answer
- `list_documents`     what is currently in the shared knowledge base
- `ask_knowledge_base` the guarded RAG agent: cited answer plus guardrail report

stdout carries the MCP protocol, so every tool runs with stdout redirected to stderr.
"""

import contextlib
import functools
import sys
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from etic_genai.config import get_settings
from etic_genai.core.logging import get_logger
from etic_genai.documents.indexing.knowledge_base import SharedKnowledgeBase

logger = get_logger(__name__)

mcp = FastMCP(
    "etic-knowledge-base",
    instructions=(
        "Business-requirements knowledge base. Use search_documents to fetch source passages "
        "and cite their `source` in your answer, or ask_knowledge_base for a ready, "
        "guardrailed answer with citations."
    ),
)

_knowledge_base: SharedKnowledgeBase | None = None
_agent: Any = None
_indexed = False


def _stdout_to_stderr(func: Callable) -> Callable:
    """Keep library `print` calls (e.g. in the chunker) off the protocol channel."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(sys.stderr):
            return func(*args, **kwargs)

    return wrapper


def _get_knowledge_base() -> SharedKnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = SharedKnowledgeBase.from_settings()
    return _knowledge_base


def _get_agent() -> Any:
    global _agent
    if _agent is None:
        from etic_genai.agents.rag_agent import RAGAgent

        _agent = RAGAgent(knowledge_base=_get_knowledge_base())
    return _agent


def _default_path() -> str:
    return get_settings().requirements_dir or "./requirements"


def _ensure_indexed() -> None:
    """Index the configured folder once per server process when INDEXING_ON_STARTUP is on."""
    global _indexed
    if not _indexed and get_settings().indexing_on_startup:
        _get_knowledge_base().index_directory(_default_path())
    _indexed = True


@mcp.tool()
@_stdout_to_stderr
def index_documents(path: str | None = None) -> dict:
    """Index supported files (md, txt, csv, pdf) from a directory into the shared knowledge base.

    Args:
        path: Directory to index. Defaults to REQUIREMENTS_DIR and must be inside the
            allowed root (KNOWLEDGE_BASE_ROOT, or REQUIREMENTS_DIR when unset).
    """
    global _indexed
    result = _get_knowledge_base().index_directory(path or _default_path())
    _indexed = True
    return result


@mcp.tool()
@_stdout_to_stderr
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid semantic + keyword (BM25) search over the shared requirement documents.

    Returns the most relevant passages with `source`, `section`, `content`, the fused
    rank `score` and a `relevance` (0-1 cosine similarity). Cite the `source` of
    passages you use.
    """
    _ensure_indexed()
    return _get_knowledge_base().search(query, top_k)


@mcp.tool()
@_stdout_to_stderr
def list_documents() -> list[dict]:
    """List the documents in the shared knowledge base with their chunk counts."""
    _ensure_indexed()
    return _get_knowledge_base().list_documents()


@mcp.tool()
@_stdout_to_stderr
def ask_knowledge_base(question: str) -> dict:
    """Answer a question about the business requirements using the guarded RAG agent.

    The answer cites its sources and has passed prompt-injection, PII, citation and
    groundedness checks. The result includes the guardrail events and scores.
    """
    _ensure_indexed()
    from etic_genai.agents.rag_agent import RAGAgent

    return RAGAgent.summarize(_get_agent().invoke(question))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
