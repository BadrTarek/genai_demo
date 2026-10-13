"""Index, search, or serve the knowledge base.

Examples:
    python run_script.py index --path requirements
    python run_script.py search --query "What are the user roles?"
    python run_script.py mcp
"""

import argparse
import json
import os

from dotenv import load_dotenv

from etic_genai.documents.indexing.knowledge_base import KnowledgeBase

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge-base indexing and MCP server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Chunk, embed, and index a directory")
    index_parser.add_argument(
        "--path", required=True, help="Directory containing source documents"
    )
    index_parser.add_argument(
        "--output", default="output/chunks.json", help="Chunk metadata output"
    )

    search_parser = subparsers.add_parser("search", help="Search indexed document chunks")
    search_parser.add_argument("--query", required=True, help="Natural-language search query")
    search_parser.add_argument("--top-k", type=int, default=5)

    subparsers.add_parser("mcp", help="Start the MCP server over stdio")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "mcp":
        from etic_genai.mcp_server import main as run_mcp_server

        run_mcp_server()
        return

    knowledge_base = KnowledgeBase(
        qdrant_path=os.getenv("QDRANT_PATH", "./data/qdrant"),
        qdrant_url=os.getenv("QDRANT_URL") or None,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "business_requirements"),
        embedding_model=os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
    if args.command == "index":
        result = knowledge_base.index_path(args.path, args.output)
    else:
        result = knowledge_base.search(args.query, args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()