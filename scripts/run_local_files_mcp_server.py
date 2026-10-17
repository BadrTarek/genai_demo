"""Run the read-only local-files MCP server over SSE on port 8001."""

from etic_genai.mcp.local_files_server import create_local_files_server


def main() -> None:
    create_local_files_server().run(transport="sse")


if __name__ == "__main__":
    main()