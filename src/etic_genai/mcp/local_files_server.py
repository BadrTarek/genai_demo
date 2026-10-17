"""MCP server for explicitly approved, read-only access to one local folder."""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException

from etic_genai.config import get_settings
from etic_genai.core.exceptions import ConfigurationError

MAX_LIST_LIMIT = 200
MAX_READ_BYTES = 1_000_000


def _root_directory() -> Path:
    configured_root = get_settings().local_files_root
    if not configured_root:
        raise ToolException("Set LOCAL_FILES_ROOT to the one folder this server may access.")

    root = Path(configured_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ToolException("LOCAL_FILES_ROOT must point to an existing directory.")
    return root


def _resolve_inside_root(root: Path, relative_path: str) -> Path:
    requested_path = Path(relative_path or ".")
    if requested_path.is_absolute() or ".." in requested_path.parts:
        raise ToolException("Path must stay inside LOCAL_FILES_ROOT.")

    resolved_path = (root / requested_path).resolve(strict=True)
    if not resolved_path.is_relative_to(root):
        raise ToolException("Path must stay inside LOCAL_FILES_ROOT.")
    return resolved_path


def list_local_files(
    approved: bool,
    relative_path: str = "",
    limit: int = 100,
) -> str:
    """List one directory inside the configured root after explicit approval."""
    if not approved:
        return "Access denied. No local files were listed."
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise ToolException(f"limit must be between 1 and {MAX_LIST_LIMIT}.")

    root = _root_directory()
    directory = _resolve_inside_root(root, relative_path)
    if not directory.is_dir():
        raise ToolException("The requested path is not a directory.")

    entries = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if entry.is_symlink():
            continue
        item = {"name": entry.name, "type": "directory" if entry.is_dir() else "file"}
        if entry.is_file():
            item["size"] = entry.stat().st_size
        entries.append(item)
        if len(entries) >= limit:
            break
    return json.dumps({"path": relative_path or ".", "entries": entries}, indent=2)


def read_local_file(approved: bool, path: str) -> str:
    """Read a UTF-8 text file inside the configured root after explicit approval."""
    if not approved:
        return "Access denied. No local file was read."

    root = _root_directory()
    file_path = _resolve_inside_root(root, path)
    if not file_path.is_file():
        raise ToolException("The requested path is not a regular file.")
    if file_path.stat().st_size > MAX_READ_BYTES:
        raise ToolException(f"File exceeds the {MAX_READ_BYTES}-byte read limit.")

    content = file_path.read_bytes()
    if len(content) > MAX_READ_BYTES:
        raise ToolException(f"File exceeds the {MAX_READ_BYTES}-byte read limit.")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolException("The requested file is not valid UTF-8 text.") from exc


def create_local_files_server(name: str = "local-files") -> Any:
    """Create a separate MCP server for local read-only file access."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise ConfigurationError(
            "MCP support is not installed. Install it with: "
            'python -m pip install -e ".[mcp]"'
        ) from exc

    mcp = MCPServer(name)

    @mcp.tool()
    def list_local_files_tool(
        approved: bool,
        relative_path: str = "",
        limit: int = 100,
    ) -> str:
        """List files in the configured local folder after the user approves."""
        return list_local_files(approved, relative_path, limit)

    @mcp.tool()
    def read_local_file_tool(approved: bool, path: str) -> str:
        """Read a UTF-8 text file in the configured local folder after approval."""
        return read_local_file(approved, path)

    return mcp