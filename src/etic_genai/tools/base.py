"""Base class for tools.

For a trivial, stateless tool, LangChain's `@tool` decorator is fine. Subclass
`BaseAgentTool` when the tool needs configuration or clients (API clients, DB
connections), or when you want consistent logging and error handling.
"""

from abc import abstractmethod
from typing import Any

from langchain_core.tools import BaseTool, ToolException

from etic_genai.core.logging import get_logger

logger = get_logger(__name__)


class BaseAgentTool(BaseTool):
    """Subclasses set `name`, `description`, `args_schema` and implement `_execute`.

    Unexpected exceptions are turned into `ToolException`, and
    `handle_tool_error=True` sends the message back to the LLM instead of crashing
    the graph, so the model can recover.
    """

    handle_tool_error: bool = True

    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """Tool logic. Receives the validated fields of `args_schema` as keyword arguments."""

    def _run(self, **kwargs: Any) -> Any:
        logger.debug("Running tool %s with %s", self.name, kwargs)
        try:
            return self._execute(**kwargs)
        except ToolException:
            raise
        except Exception as exc:
            logger.exception("Tool %s failed", self.name)
            raise ToolException(f"{self.name} failed: {exc}") from exc
