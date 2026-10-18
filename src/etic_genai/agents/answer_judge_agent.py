"""Agent that scores whether an assistant response is correct and evidence-based."""

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from etic_genai.agents.base import BaseAgent
from etic_genai.nodes import LLMNode
from etic_genai.prompts import AnswerJudgePrompt


class AnswerJudgment(BaseModel):
    score: Literal[0, 1]
    reasoning: str


class AnswerJudgeAgent(BaseAgent):
    name = "answer_judge"
    description = "Evaluates assistant response correctness against retrieved evidence."

    def default_tools(self) -> list[BaseTool]:
        return []

    def build_graph(self) -> StateGraph:
        prompt = AnswerJudgePrompt().to_chat_prompt()
        llm_node = LLMNode(self.llm, prompt, output_schema=AnswerJudgment)

        builder = StateGraph(self.state_schema)
        builder.add_node(llm_node.name, llm_node)
        builder.add_edge(START, llm_node.name)
        builder.add_edge(llm_node.name, END)
        return builder

    def judge(
        self,
        question: str,
        answer: str,
        context: str,
        reference_answer: str,
    ) -> AnswerJudgment:
        evaluation_input = (
            f"Question:\n{question}\n\n"
            f"Assistant answer:\n{answer}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Reference answer:\n{reference_answer}"
        )
        result = self.invoke({"messages": [HumanMessage(content=evaluation_input)]})
        content = result["messages"][-1].content
        if not isinstance(content, str):
            raise ValueError("Answer judge returned a non-text response.")
        try:
            return AnswerJudgment.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError("Answer judge returned invalid JSON or an invalid score.") from exc
