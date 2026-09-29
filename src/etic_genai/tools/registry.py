"""Registry so agents can look tools up by name instead of importing them directly."""

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from etic_genai.core.exceptions import ToolNotFoundError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"Unknown tool: {name}") from None

    def get_many(self, names: Iterable[str]) -> list[BaseTool]:
        return [self.get(name) for name in names]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


tool_registry = ToolRegistry()
