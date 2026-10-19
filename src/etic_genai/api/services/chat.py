"""Document retrieval and agent invocation for chat requests."""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from etic_genai.agents import AssistantAgent
from etic_genai.api.dependencies import IngestionServices
from etic_genai.api.schemas import RetrievedSource
from etic_genai.auth.store import AuthUser

NO_DOCUMENT_CONTEXT = "No matching uploaded document excerpts were found."


@dataclass(frozen=True)
class ChatAgentResult:
    answer: str
    context: str
    sources: list[RetrievedSource]


def run_chat_agent(
    message: str,
    conversation_id: str,
    current_user: AuthUser,
    agent: AssistantAgent,
    services: IngestionServices,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
) -> ChatAgentResult:
    """Retrieve owned documents, then public fallback, before invoking the agent."""
    query_vector = services.embedding_model.embed_query(message)
    matches = services.qdrant_store.search(
        query_vector,
        query_text=message,
        limit=5,
        owner_id=current_user.user_id,
    )


    context_parts: list[str] = []
    sources: list[RetrievedSource] = []
    for index, match in enumerate(matches, start=1):
        payload = match.payload or {}
        content = str(payload.get("content") or "")
        if not content.strip():
            continue

        title = str(payload.get("document_title") or "Untitled document")
        source = str(payload.get("source") or payload.get("document_id") or "Unknown source")
        section = payload.get("section")
        context_parts.append(
            f"[{index}] Document: {title}; source: {source}; section: {section or 'N/A'}\n{content}"
        )
        sources.append(
            RetrievedSource(
                document_title=title,
                source=source,
                section=section,
                score=float(match.score),
            )
        )

    context = "\n\n".join(context_parts) or NO_DOCUMENT_CONTEXT
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "retrieved_context": context,
        },
        thread_id=f"{current_user.user_id}:{conversation_id}",
        callbacks=callbacks,
    )
    return ChatAgentResult(
        answer=str(result["messages"][-1].content),
        context=context,
        sources=sources,
    )
