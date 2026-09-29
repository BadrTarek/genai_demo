import pytest

from etic_genai.core.exceptions import ToolNotFoundError
from etic_genai.tools import CurrentDateTimeTool, ToolRegistry


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
