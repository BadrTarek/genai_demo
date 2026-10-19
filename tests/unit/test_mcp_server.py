from pathlib import Path

import pytest

from etic_genai import mcp_server


@pytest.fixture
def server(monkeypatch, knowledge_base, requirements_dir: Path):
    monkeypatch.setattr(mcp_server, "_knowledge_base", knowledge_base)
    monkeypatch.setattr(mcp_server, "_indexed", False)
    monkeypatch.setenv("REQUIREMENTS_DIR", str(requirements_dir))
    mcp_server.get_settings.cache_clear()
    yield mcp_server
    mcp_server.get_settings.cache_clear()


def test_search_indexes_configured_folder_on_first_call(server):
    hits = server.search_documents("Who can reopen approved invoices?", top_k=1)

    assert hits[0]["source"] == "user_roles.csv"
    assert {d["document_id"] for d in server.list_documents()} == {
        "authentication.md",
        "invoice_approval.txt",
        "user_roles.csv",
    }


def test_index_documents_tool(server, requirements_dir: Path):
    report = server.index_documents(str(requirements_dir))

    assert report["documents_indexed"] == 3


def test_tools_keep_prints_off_stdout(server, requirements_dir: Path, capsys):
    # The chunker prints document titles; stdout is the MCP protocol channel.
    server.index_documents(str(requirements_dir))

    assert capsys.readouterr().out == ""


async def test_tools_are_registered_with_their_parameters(server):
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}

    assert set(tools) == {
        "index_documents",
        "search_documents",
        "list_documents",
        "ask_knowledge_base",
    }
    assert set(tools["search_documents"].inputSchema["properties"]) == {"query", "top_k"}
