from etic_genai.tools.base import BaseAgentTool
from etic_genai.tools.common.datetime_tool import CurrentDateTimeTool
from etic_genai.tools.document_access import (
    DocumentAccessRequestTool,
    SharePointAuthFinishTool,
    SharePointAuthStartTool,
    SharePointListFilesTool,
    SharePointSearchTool,
    SharePointTenantInfoTool
)
from etic_genai.tools.registry import ToolRegistry, tool_registry

# Register shared tools here so agents can request them by name.
tool_registry.register(CurrentDateTimeTool())
tool_registry.register(DocumentAccessRequestTool())
tool_registry.register(SharePointListFilesTool())
tool_registry.register(SharePointSearchTool())
tool_registry.register(SharePointAuthStartTool())
tool_registry.register(SharePointAuthFinishTool())
tool_registry.register(SharePointTenantInfoTool())
__all__ = [
    "BaseAgentTool",
    "CurrentDateTimeTool",
    "DocumentAccessRequestTool",
    "SharePointListFilesTool",
    "SharePointSearchTool",
    "SharePointAuthStartTool",
    "SharePointAuthFinishTool",
    "SharePointTenantInfoTool",
    "ToolRegistry",
    "tool_registry",
]
