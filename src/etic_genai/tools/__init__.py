from etic_genai.tools.base import BaseAgentTool
from etic_genai.tools.common.datetime_tool import CurrentDateTimeTool
from etic_genai.tools.registry import ToolRegistry, tool_registry

# Register shared tools here so agents can request them by name.
tool_registry.register(CurrentDateTimeTool())

__all__ = ["BaseAgentTool", "CurrentDateTimeTool", "ToolRegistry", "tool_registry"]
