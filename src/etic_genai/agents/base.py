"""Base class for agents.

An agent owns its LLM, tools, prompt and LangGraph graph. Subclasses only
describe the graph in `build_graph`; compiling, memory and the invoke/stream
API are shared here.
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from etic_genai.core.llm import get_llm
from etic_genai.core.logging import get_logger
from etic_genai.graphs.state import BaseAgentState

AgentInput = str | dict[str, Any]


class BaseAgent(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    state_schema: ClassVar[type] = BaseAgentState

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: Sequence[BaseTool] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.logger = get_logger(f"{__name__}.{self.name}")
        self.llm = llm or get_llm()
        self.tools = list(tools) if tools is not None else self.default_tools()
        self.checkpointer = checkpointer
        self._graph: CompiledStateGraph | None = None

    def default_tools(self) -> list[BaseTool]:
        """Tools used when none are passed to the constructor."""
        return []

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Return the (uncompiled) graph. Use `StateGraph(self.state_schema)`."""

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            self._graph = self.build_graph().compile(checkpointer=self.checkpointer, name=self.name)
        return self._graph

    def invoke(self, input: AgentInput, thread_id: str | None = None, **kwargs: Any) -> dict:
        return self.graph.invoke(self._to_state(input), self._config(thread_id), **kwargs)

    async def ainvoke(self, input: AgentInput, thread_id: str | None = None, **kwargs: Any) -> dict:
        return await self.graph.ainvoke(self._to_state(input), self._config(thread_id), **kwargs)

    def stream(self, input: AgentInput, thread_id: str | None = None, **kwargs: Any) -> Iterator:
        yield from self.graph.stream(self._to_state(input), self._config(thread_id), **kwargs)

    async def astream(
        self, input: AgentInput, thread_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator:
        async for chunk in self.graph.astream(
            self._to_state(input), self._config(thread_id), **kwargs
        ):
            yield chunk

    @staticmethod
    def _to_state(input: AgentInput) -> dict[str, Any]:
        if isinstance(input, str):
            return {"messages": [HumanMessage(content=input)]}
        return input

    def _config(self, thread_id: str | None) -> RunnableConfig:
        # A thread_id is required whenever a checkpointer is set.
        if self.checkpointer is not None and thread_id is None:
            thread_id = str(uuid.uuid4())
        configurable = {"thread_id": thread_id} if thread_id else {}
        return {"configurable": configurable, "run_name": self.name}
