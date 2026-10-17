import pytest

from etic_genai.core.exceptions import ToolNotFoundError
from etic_genai.tools import CurrentDateTimeTool, DocumentAccessRequestTool, SharePointListFilesTool
from etic_genai.tools.registry import ToolRegistry


def test_datetime_tool_returns_iso_string():
    result = CurrentDateTimeTool().invoke({"timezone": "UTC"})
    assert result.endswith("+00:00")


def test_tool_errors_are_returned_to_the_llm():
    result = CurrentDateTimeTool().invoke({"timezone": "Not/AZone"})
    assert "current_datetime failed" in result


def test_registry():
    registry = ToolRegistry()
    tool = registry.register(CurrentDateTimeTool())
    assert registry.get("current_datetime") is tool
    with pytest.raises(ValueError):
        registry.register(CurrentDateTimeTool())
    with pytest.raises(ToolNotFoundError):
        registry.get("missing")


def test_document_access_request_requires_user_approval():
    result = DocumentAccessRequestTool().invoke(
        {"provider": "sharepoint", "purpose": "List files for the current task."}
    )
    assert "approval_required" in result
    assert "sharepoint" in result


def test_sharepoint_tool_does_not_access_files_when_denied():
    result = SharePointListFilesTool().invoke(
        {"approved": False, "site_id": "contoso.sharepoint.com,site-id"}
    )
    assert "Access denied by user" in result
