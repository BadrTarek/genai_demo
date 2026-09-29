from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from etic_genai.nodes.base import BaseNode


class LLMNode(BaseNode):
    """Runs `prompt | llm` (with tools bound) and appends the reply to `messages`."""

    name = "llm"

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        super().__init__()
        model = llm.bind_tools(tools) if tools else llm
        self.chain = prompt | model

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        response = self.chain.invoke(state, config)
        return {"messages": [response]}
