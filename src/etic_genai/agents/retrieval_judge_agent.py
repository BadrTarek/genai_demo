"""Agent that scores whether retrieved context is useful for a question."""

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from etic_genai.agents.base import BaseAgent
from etic_genai.nodes import LLMNode
from etic_genai.prompts import RetrievalJudgePrompt


class RetrievalJudgment(BaseModel):
    score: Literal[0, 1]
    reasoning: str


class RetrievalJudgeAgent(BaseAgent):
    name = "retrieval_judge"
    description = "Evaluates whether retrieved document context is relevant to a question."

    def default_tools(self) -> list[BaseTool]:
        return []

    def build_graph(self) -> StateGraph:
        prompt = RetrievalJudgePrompt().to_chat_prompt()
        llm_node = LLMNode(self.llm, prompt)

        builder = StateGraph(self.state_schema)
        builder.add_node(llm_node.name, llm_node)
        builder.add_edge(START, llm_node.name)
        builder.add_edge(llm_node.name, END)
        return builder

    def judge(self, question: str, context: str) -> RetrievalJudgment:
        result = self.invoke(
            {
                "messages": [
                    HumanMessage(content=f"Question:\n{question}\n\nRetrieved context:\n{context}")
                ]
            }
        )
        content = result["messages"][-1].content
        if not isinstance(content, str):
            raise ValueError("Retrieval judge returned a non-text response.")
        try:
            return RetrievalJudgment.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError("Retrieval judge returned invalid JSON or an invalid score.") from exc
