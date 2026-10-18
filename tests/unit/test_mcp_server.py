from pathlib import Path

import pytest

from etic_genai import mcp_server


@pytest.fixture
def server(monkeypatch, knowledge_base, requirements_dir: Path):
    monkeypatch.setattr(mcp_server, "_knowledge_base", knowledge_base)
    monkeypatch.setattr(mcp_server, "_indexed", False)
    monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(requirements_dir))
    mcp_server.get_settings.cache_clear()
    yield mcp_server
    mcp_server.get_settings.cache_clear()


def test_search_indexes_configured_folder_on_first_call(server):
    hits = server.search_documents("Who can reopen approved invoices?", top_k=1)

    assert hits[0]["source"] == "user_roles.csv"
    assert set(hits[0]) == {"score", "source", "document_title", "section", "content"}
    assert {d["document_id"] for d in server.list_documents()} == {
        "authentication.md",
        "invoice_approval.txt",
        "user_roles.csv",
    }


def test_index_documents_tool(server, requirements_dir: Path):
    report = server.index_documents(str(requirements_dir))

    assert report["documents_indexed"] == 3


async def test_tools_are_registered(server):
    names = {tool.name for tool in await server.mcp.list_tools()}

    assert names == {"index_documents", "search_documents", "list_documents", "ask_knowledge_base"}
