"""Run the project MCP server: `python scripts/run_mcp_server.py`."""

from etic_genai.mcp import create_mcp_server


mcp = create_mcp_server()


def main() -> None:
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
