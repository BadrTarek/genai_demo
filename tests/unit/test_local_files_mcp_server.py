import json
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException

from etic_genai.mcp import local_files_server


def test_local_file_tools_require_approval(monkeypatch):
    monkeypatch.setattr(
        local_files_server,
        "get_settings",
        lambda: SimpleNamespace(local_files_root=None),
    )

    assert local_files_server.list_local_files(False) == "Access denied. No local files were listed."
    assert local_files_server.read_local_file(False, "secret.txt") == (
        "Access denied. No local file was read."
    )


def test_list_and_read_approved_files_within_root(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("private notes", encoding="utf-8")
    monkeypatch.setattr(
        local_files_server,
        "get_settings",
        lambda: SimpleNamespace(local_files_root=str(tmp_path)),
    )

    listing = json.loads(local_files_server.list_local_files(True))

    assert listing["entries"] == [{"name": "notes.txt", "type": "file", "size": 13}]
    assert local_files_server.read_local_file(True, "notes.txt") == "private notes"


def test_local_file_tools_reject_parent_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_files_server,
        "get_settings",
        lambda: SimpleNamespace(local_files_root=str(tmp_path)),
    )

    with pytest.raises(ToolException, match="inside LOCAL_FILES_ROOT"):
        local_files_server.read_local_file(True, "../outside.txt")