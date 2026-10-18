from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from etic_genai.nodes.base import BaseNode


class LLMNode(BaseNode):
    """Runs `prompt | llm` (with tools bound) and appends the reply to `messages`."""

    name = "llm"

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        tools: Sequence[BaseTool] = (),
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        super().__init__()
        self.output_schema = output_schema
        if output_schema is not None:
            model = llm.with_structured_output(output_schema, method="json_schema")
        else:
            model = llm.bind_tools(tools) if tools else llm
        self.chain = prompt | model

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        prompt_state = {
            **state,
            "retrieved_context": state.get("retrieved_context", "No document excerpts provided."),
        }
        response = self.chain.invoke(prompt_state, config)
        if self.output_schema is not None:
            if not isinstance(response, self.output_schema):
                raise TypeError(
                    f"Structured LLM output must be {self.output_schema.__name__}, "
                    f"got {type(response).__name__}."
                )
            response = AIMessage(content=response.model_dump_json())
        return {"messages": [response]}
