"""MCP server exposing the knowledge base to MCP clients such as GitHub Copilot.

Tools:
- `index_documents`   index (incrementally) a directory under the allowed root
- `search_documents`  raw semantic search; the client's own model writes the answer
- `list_documents`    what is currently indexed
- `ask_knowledge_base` the guarded RAG agent: cited answer + guardrail and eval report

stdout carries the MCP protocol, so this process must never `print`; logs go to stderr.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from etic_genai.config import get_settings
from etic_genai.core.logging import get_logger
from etic_genai.documents.indexing.knowledge_base import KnowledgeBase

logger = get_logger(__name__)

mcp = FastMCP(
    "etic-knowledge-base",
    instructions=(
        "Business-requirements knowledge base. Use search_documents to fetch source passages "
        "and cite their `source` in your answer, or ask_knowledge_base for a ready, "
        "guardrailed answer with citations."
    ),
)

_knowledge_base: KnowledgeBase | None = None
_agent: Any = None
_indexed = False


def _get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase.from_settings()
    return _knowledge_base


def _get_agent() -> Any:
    global _agent
    if _agent is None:
        from etic_genai.agents.rag_agent import RAGAgent

        _agent = RAGAgent(knowledge_base=_get_knowledge_base())
    return _agent


def _ensure_indexed() -> None:
    """Index the configured folder once per server process. Incremental, so cheap."""
    global _indexed
    if not _indexed and get_settings().indexing_on_startup:
        _get_knowledge_base().index_path(get_settings().knowledge_base_path)
    _indexed = True


@mcp.tool()
def index_documents(path: str | None = None, force: bool = False) -> dict:
    """Index supported files (md, txt, csv, pdf, docx) from a directory into the knowledge base.

    Args:
        path: Directory to index. Defaults to the configured knowledge-base folder and must be
            inside the allowed root.
        force: Re-embed every document even if it has not changed.
    """
    global _indexed
    result = _get_knowledge_base().index_path(
        path or get_settings().knowledge_base_path, force=force
    )
    _indexed = True
    return result


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over the indexed requirement documents.

    Returns the most relevant passages with `source`, `section`, `content` and a
    similarity `score` (0-1). Cite the `source` of passages you use.
    """
    _ensure_indexed()
    return [
        {key: hit.get(key) for key in ("score", "source", "document_title", "section", "content")}
        for hit in _get_knowledge_base().search(query, top_k)
    ]


@mcp.tool()
def list_documents() -> list[dict]:
    """List the documents currently in the knowledge base with their chunk counts."""
    _ensure_indexed()
    return _get_knowledge_base().list_documents()


@mcp.tool()
def ask_knowledge_base(question: str) -> dict:
    """Answer a question about the business requirements using the guarded RAG agent.

    The answer cites its sources and has passed prompt-injection, PII, citation and
    groundedness checks. The result includes the guardrail events and evaluation scores.
    """
    _ensure_indexed()
    from etic_genai.agents.rag_agent import RAGAgent

    return RAGAgent.summarize(_get_agent().invoke(question))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
