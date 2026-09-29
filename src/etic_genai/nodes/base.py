"""Base class for graph nodes.

A node takes the current state and returns a partial state update (a dict).
Writing nodes as classes lets them carry dependencies (LLM, tools, clients)
and keeps them testable in isolation.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from langchain_core.runnables import RunnableConfig

from etic_genai.core.logging import get_logger


class BaseNode(ABC):
    """Subclasses set `name` and implement `run`. Add the instance to a graph with
    `builder.add_node(node.name, node)`.
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self.logger = get_logger(f"{__name__}.{self.name}")

    @abstractmethod
    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        """Return the state keys to update."""

    def __call__(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        self.logger.debug("Entering node %s", self.name)
        return self.run(state, config)
