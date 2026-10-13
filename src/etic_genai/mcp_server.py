"""MCP server exposing knowledge-base indexing and vector search tools."""

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from etic_genai.documents.indexing.knowledge_base import KnowledgeBase

mcp = FastMCP("etic-knowledge-base")
_knowledge_base: KnowledgeBase | None = None
load_dotenv()


def _get_knowledge_base() -> KnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(
            qdrant_path=os.getenv("QDRANT_PATH", "./data/qdrant"),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "business_requirements"),
            allowed_root=os.getenv("KNOWLEDGE_BASE_ROOT"),
            embedding_model=os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        )
    return _knowledge_base


@mcp.tool()
def index_knowledge_base(path: str, output_path: str | None = None) -> dict:
    """Index supported files from a user-provided directory into Qdrant."""
    return _get_knowledge_base().index_path(path, output_path)


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """Search indexed document chunks using semantic similarity."""
    return _get_knowledge_base().search(query, top_k)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()