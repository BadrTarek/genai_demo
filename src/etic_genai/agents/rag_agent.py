"""Guarded RAG agent over the requirements knowledge base.

START → input_guardrails ─┬─(blocked)──────────────────────────────┐
                          └→ retrieve ─┬─(no_context / error)──────┤
                                       └→ generate → output_guardrails
                                            ↑            │
                                            └──(retry)───┤
                                                         ├─(ok)→ evaluate ─┐
                                                         └─(ungrounded)────┤
                                                                           ↓
                                                                finalize → END
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from etic_genai.agents.base import BaseAgent
from etic_genai.config import Settings, get_settings
from etic_genai.documents.indexing.knowledge_base import KnowledgeBase
from etic_genai.evaluation import AnswerRelevanceEvaluator, Evaluator, FaithfulnessEvaluator
from etic_genai.graphs.state import RAGState
from etic_genai.guardrails import (
    CitationGuardrail,
    GroundednessGuardrail,
    GuardrailPipeline,
    InputLengthGuardrail,
    PIIRedactionGuardrail,
    PromptInjectionGuardrail,
)
from etic_genai.nodes.rag import (
    EvaluateNode,
    FinalizeNode,
    GenerateNode,
    InputGuardrailsNode,
    OutputGuardrailsNode,
    RetrieveNode,
)


class RAGAgent(BaseAgent):
    name = "requirements_assistant"
    description = "Answers questions about business requirements with cited, grounded answers."
    state_schema = RAGState

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        llm: BaseChatModel | None = None,
        judge_llm: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        settings: Settings | None = None,
        input_guardrails: GuardrailPipeline | None = None,
        output_guardrails: GuardrailPipeline | None = None,
        online_evaluators: list[Evaluator] | None = None,
    ) -> None:
        super().__init__(llm=llm, tools=[], checkpointer=checkpointer)
        self.settings = settings or get_settings()
        self.knowledge_base = knowledge_base or KnowledgeBase.from_settings(self.settings)
        # A separate (possibly cheaper) judge model can be passed; default is the same LLM.
        self.judge_llm = judge_llm or self.llm
        self.input_guardrails = input_guardrails or self.default_input_guardrails()
        self.output_guardrails = output_guardrails or self.default_output_guardrails()
        if online_evaluators is None:
            online_evaluators = (
                [AnswerRelevanceEvaluator(self.judge_llm)]
                if self.settings.online_evaluation
                else []
            )
        self.online_evaluators = online_evaluators

    def default_input_guardrails(self) -> GuardrailPipeline:
        return GuardrailPipeline(
            [
                InputLengthGuardrail(max_chars=self.settings.max_question_chars),
                PromptInjectionGuardrail(),
                PIIRedactionGuardrail(),
            ]
        )

    def default_output_guardrails(self) -> GuardrailPipeline:
        return GuardrailPipeline(
            [
                PIIRedactionGuardrail(),
                CitationGuardrail(),
                GroundednessGuardrail(
                    FaithfulnessEvaluator(self.judge_llm),
                    threshold=self.settings.groundedness_threshold,
                ),
            ]
        )

    def build_graph(self) -> StateGraph:
        nodes = [
            InputGuardrailsNode(self.input_guardrails),
            RetrieveNode(
                self.knowledge_base,
                top_k=self.settings.max_search_results,
                min_score=self.settings.min_relevance_score,
            ),
            GenerateNode(self.llm, agent_name=self.name),
            OutputGuardrailsNode(
                self.output_guardrails, max_attempts=self.settings.max_generation_attempts
            ),
            EvaluateNode(self.online_evaluators),
            FinalizeNode(),
        ]
        builder = StateGraph(self.state_schema)
        for node in nodes:
            builder.add_node(node.name, node)

        builder.add_edge(START, "input_guardrails")
        builder.add_conditional_edges(
            "input_guardrails",
            lambda s: "finalize" if s["status"] == "blocked" else "retrieve",
            ["retrieve", "finalize"],
        )
        builder.add_conditional_edges(
            "retrieve",
            lambda s: "generate" if s["status"] == "ok" else "finalize",
            ["generate", "finalize"],
        )
        builder.add_edge("generate", "output_guardrails")
        builder.add_conditional_edges(
            "output_guardrails", self._route_after_output, ["generate", "evaluate", "finalize"]
        )
        builder.add_edge("evaluate", "finalize")
        builder.add_edge("finalize", END)
        return builder

    def _route_after_output(self, state: dict[str, Any]) -> str:
        if state["status"] == "retry":
            return "generate"
        if state["status"] == "ok" and self.online_evaluators:
            return "evaluate"
        return "finalize"

    @staticmethod
    def summarize(state: dict[str, Any]) -> dict[str, Any]:
        """A JSON-friendly view of a finished turn (used by the CLI and MCP server)."""
        return {
            "answer": state["messages"][-1].content,
            "status": state.get("status"),
            "attempts": state.get("attempts", 0),
            "sources": [
                {
                    "source": c.get("source"),
                    "section": c.get("section"),
                    "score": c.get("score"),
                }
                for c in state.get("contexts", [])
            ],
            "guardrail_events": state.get("guardrail_events", []),
            "evaluations": state.get("evaluations", {}),
        }
