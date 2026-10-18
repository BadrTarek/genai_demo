"""Conversational document retrieval endpoint."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends

from etic_genai.agents import AssistantAgent
from etic_genai.api.dependencies import (
    IngestionServices,
    get_agent,
    get_current_user,
    get_ingestion_services,
    get_usage_store,
)
from etic_genai.api.schemas import ChatRequest, ChatResponse
from etic_genai.api.services.chat import run_chat_agent
from etic_genai.auth.store import AuthUser
from etic_genai.usage.callback import UsageTracker
from etic_genai.usage.store import UsageStore

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the assistant",
    description=(
        "Omit conversation_id to create a new conversation. Use the returned ID "
        "on later requests to continue its history. "
        "The question is embedded and matching uploaded document chunks are supplied as context."
    ),
    responses={401: {"description": "A valid bearer token is required."}},
)
def chat(
    request: ChatRequest,
    agent: Annotated[AssistantAgent, Depends(get_agent)],
    services: Annotated[IngestionServices, Depends(get_ingestion_services)],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    usage_store: Annotated[UsageStore, Depends(get_usage_store)],
) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    usage = UsageTracker()
    try:
        result = run_chat_agent(
            message=request.message,
            conversation_id=conversation_id,
            current_user=current_user,
            agent=agent,
            services=services,
            callbacks=[usage],
        )
    finally:
        usage_store.add_usage(current_user.user_id, conversation_id, usage)
    return ChatResponse(
        conversation_id=conversation_id,
        response=result.answer,
        sources=result.sources,
    )
