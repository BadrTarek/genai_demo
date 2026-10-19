"""Guarded RAG agent over the shared knowledge base (hybrid search).

START → input_guardrails ─┬─(blocked)──────────────────────────────┐
                          └→ retrieve ─┬─(no_context / error)──────┤
                                       └→ generate → output_guardrails
                                            ↑            │
                                            └──(retry)───┤
                                                         ├─(ok / ungrounded)─┐
                                                                             ↓
                                                                  finalize → END
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from etic_genai.agents.base import BaseAgent
from etic_genai.config import Settings, get_settings
from etic_genai.documents.indexing.knowledge_base import SharedKnowledgeBase
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
        knowledge_base: SharedKnowledgeBase | None = None,
        llm: BaseChatModel | None = None,
        judge_llm: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        settings: Settings | None = None,
        input_guardrails: GuardrailPipeline | None = None,
        output_guardrails: GuardrailPipeline | None = None,
    ) -> None:
        super().__init__(llm=llm, tools=[], checkpointer=checkpointer)
        self.settings = settings or get_settings()
        self.knowledge_base = knowledge_base or SharedKnowledgeBase.from_settings()
        # A separate (possibly cheaper) judge model can be passed; default is the same LLM.
        self.judge_llm = judge_llm or self.llm
        self.input_guardrails = input_guardrails or self.default_input_guardrails()
        self.output_guardrails = output_guardrails or self.default_output_guardrails()

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
                    self.judge_llm, threshold=self.settings.groundedness_threshold
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
            "output_guardrails",
            lambda s: "generate" if s["status"] == "retry" else "finalize",
            ["generate", "finalize"],
        )
        builder.add_edge("finalize", END)
        return builder

    @staticmethod
    def summarize(state: dict[str, Any]) -> dict[str, Any]:
        """A JSON-friendly view of a finished turn (used by the MCP server)."""
        return {
            "answer": state["messages"][-1].content,
            "status": state.get("status"),
            "attempts": state.get("attempts", 0),
            "sources": [
                {
                    "source": c.get("source"),
                    "section": c.get("section"),
                    "relevance": c.get("relevance"),
                }
                for c in state.get("contexts", [])
            ],
            "guardrail_events": state.get("guardrail_events", []),
            "evaluations": state.get("evaluations", {}),
        }
