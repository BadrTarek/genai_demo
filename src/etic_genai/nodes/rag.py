"""Nodes of the RAG agent graph. Each one reads the state and returns only the keys it changes."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from etic_genai.documents.indexing.knowledge_base import KnowledgeBase
from etic_genai.evaluation.base import EvalSample, Evaluator
from etic_genai.guardrails import (
    NOT_FOUND_ANSWER,
    GuardrailContext,
    GuardrailPipeline,
    cited_indices,
)
from etic_genai.nodes.base import BaseNode
from etic_genai.prompts import RAGAnswerPrompt

BLOCKED_ANSWER = "I can't help with that request. {reason}"
ERROR_ANSWER = "The knowledge base is currently unavailable. Please try again later."
UNGROUNDED_ANSWER = (
    "I couldn't produce an answer that is fully supported by the knowledge base. "
    "These passages may contain what you are looking for:"
)


def latest_question(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def describe_source(context: dict[str, Any]) -> str:
    source = context.get("source") or context.get("document_id", "unknown")
    section = context.get("section")
    return f"{source} > {section}" if section else source


def format_passages(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {describe_source(c)})\n{c.get('content', '')}"
        for i, c in enumerate(contexts, start=1)
    )


class InputGuardrailsNode(BaseNode):
    """Validates and sanitises the question. Also resets the per-turn state."""

    name = "input_guardrails"

    def __init__(self, pipeline: GuardrailPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        result = self.pipeline.run(latest_question(state["messages"]))
        update: dict[str, Any] = {
            "question": result.text,
            "contexts": [],
            "answer": "",
            "status": "ok",
            "attempts": 0,
            "feedback": "",
            "guardrail_events": result.events("input"),
            "evaluations": {},
        }
        if not result.passed:
            self.logger.info("Input blocked by %s", result.failure.name)
            update["status"] = "blocked"
            update["answer"] = BLOCKED_ANSWER.format(reason=result.failure.reason)
        return update


class RetrieveNode(BaseNode):
    name = "retrieve"

    def __init__(
        self, knowledge_base: KnowledgeBase, top_k: int = 5, min_score: float = 0.3
    ) -> None:
        super().__init__()
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.min_score = min_score

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        try:
            hits = self.knowledge_base.search(state["question"], self.top_k)
        except Exception:
            self.logger.exception("Retrieval failed")
            return {"status": "error", "answer": ERROR_ANSWER}
        # Relevance threshold doubles as a scope guardrail: off-topic questions find nothing.
        relevant = [hit for hit in hits if hit["score"] >= self.min_score]
        self.logger.debug(
            "Retrieved %d/%d passages above %.2f", len(relevant), len(hits), self.min_score
        )
        if not relevant:
            return {"contexts": [], "status": "no_context", "answer": NOT_FOUND_ANSWER}
        return {"contexts": relevant}


class GenerateNode(BaseNode):
    name = "generate"

    def __init__(self, llm: BaseChatModel, agent_name: str, history_window: int = 6) -> None:
        super().__init__()
        self.llm = llm
        self.agent_name = agent_name
        self.prompt = RAGAnswerPrompt()
        self.history_window = history_window

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        feedback = state.get("feedback")
        system = self.prompt.render(
            agent_name=self.agent_name,
            not_found=NOT_FOUND_ANSWER,
            contexts=format_passages(state["contexts"]),
            feedback=(
                f"\nYour previous answer was rejected: {feedback} Write a corrected answer.\n"
                if feedback
                else ""
            ),
        )
        # Prior turns give the model conversational context; the current question is
        # the sanitised one from the input guardrails, not the raw user message.
        history = [m for m in state["messages"][:-1] if m.type in ("human", "ai")]
        messages = [
            SystemMessage(content=system),
            *history[-self.history_window :],
            HumanMessage(content=state["question"]),
        ]
        response = self.llm.invoke(messages, config)
        return {"answer": response.text.strip(), "attempts": state.get("attempts", 0) + 1}


class OutputGuardrailsNode(BaseNode):
    """Checks the answer; asks for a regeneration or falls back when a check fails."""

    name = "output_guardrails"

    def __init__(self, pipeline: GuardrailPipeline, max_attempts: int = 2) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.max_attempts = max_attempts

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        answer = state["answer"]
        if NOT_FOUND_ANSWER.lower() in answer.lower():
            return {"status": "no_context", "answer": NOT_FOUND_ANSWER}

        attempts = state.get("attempts", 1)
        context = GuardrailContext(
            question=state["question"], contexts=[c.get("content", "") for c in state["contexts"]]
        )
        result = self.pipeline.run(answer, context)
        events = [*state.get("guardrail_events", []), *result.events(f"output#{attempts}")]
        evaluations = dict(state.get("evaluations", {}))
        for check in result.results:
            if check.name == "groundedness" and "score" in check.metadata:
                evaluations["groundedness"] = {
                    "score": check.metadata["score"],
                    "passed": check.passed,
                    "reason": check.reason,
                }

        update: dict[str, Any] = {"guardrail_events": events, "evaluations": evaluations}
        if result.passed:
            update.update(status="ok", answer=result.text, feedback="")
        elif attempts < self.max_attempts:
            self.logger.info("Answer rejected by %s, regenerating", result.failure.name)
            update.update(status="retry", feedback=result.failure.reason)
        else:
            self.logger.warning("Answer rejected by %s, giving up", result.failure.name)
            update.update(status="ungrounded", answer=UNGROUNDED_ANSWER)
        return update


class EvaluateNode(BaseNode):
    """Online evaluation: scores every answered turn so quality can be monitored."""

    name = "evaluate"

    def __init__(self, evaluators: list[Evaluator]) -> None:
        super().__init__()
        self.evaluators = evaluators

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        sample = EvalSample(
            question=state["question"],
            answer=state["answer"],
            contexts=[c.get("content", "") for c in state["contexts"]],
        )
        evaluations = dict(state.get("evaluations", {}))
        for evaluator in self.evaluators:
            evaluations[evaluator.name] = evaluator.evaluate(sample).to_dict()
        return {"evaluations": evaluations}


class FinalizeNode(BaseNode):
    """Appends the user-facing reply (answer plus source list) to the conversation."""

    name = "finalize"

    def run(self, state: Any, config: RunnableConfig) -> dict[str, Any]:
        answer = state["answer"]
        contexts = state.get("contexts", [])
        status = state.get("status")
        if status == "ok":
            cited = [n for n in cited_indices(answer) if 1 <= n <= len(contexts)]
            if cited:
                # A Markdown list, so chat UIs render one source per line.
                sources = "\n".join(f"- [{n}] {describe_source(contexts[n - 1])}" for n in cited)
                answer = f"{answer}\n\nSources:\n{sources}"
        elif status == "ungrounded":
            sources = "\n".join(f"- {describe_source(c)}" for c in contexts)
            answer = f"{answer}\n\n{sources}"
        return {"messages": [AIMessage(content=answer)]}
