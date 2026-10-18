"""Shared FastAPI dependencies and service factories."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from etic_genai.agents import AssistantAgent
from etic_genai.auth.store import AuthStore, AuthUser
from etic_genai.config import get_settings
from etic_genai.documents.indexing.qdrant_store import QdrantStore
from etic_genai.documents.loaders.factory import DocumentLoaderFactory
from etic_genai.documents.processing.chunker import DocumentChunker
from etic_genai.embeddings.model import EmbeddingModel
from etic_genai.memory import get_sqlite_checkpointer
from etic_genai.usage.store import UsageStore

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class IngestionServices:
    loader_factory: DocumentLoaderFactory
    chunker: DocumentChunker
    embedding_model: EmbeddingModel
    qdrant_store: QdrantStore


@lru_cache
def get_auth_store() -> AuthStore:
    settings = get_settings()
    return AuthStore(settings.sqlite_db_path or "./data/documents.db")


@lru_cache
def get_agent() -> AssistantAgent:
    settings = get_settings()
    return AssistantAgent(
        checkpointer=get_sqlite_checkpointer(settings.sqlite_db_path or "./data/documents.db")
    )


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    auth_store: Annotated[AuthStore, Depends(get_auth_store)],
) -> AuthUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="****** required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = auth_store.get_active_session_user(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_admin_user(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> AuthUser:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access is required.",
        )
    return current_user


@lru_cache
def get_usage_store() -> UsageStore:
    settings = get_settings()
    get_auth_store()
    return UsageStore(settings.sqlite_db_path or "./data/documents.db")


@lru_cache
def get_ingestion_services() -> IngestionServices:
    settings = get_settings()
    if settings.azure_openai_endpoint:
        embedding_model = EmbeddingModel(
            model=settings.embedding_model,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.embedding_deployment_name,
            api_version=settings.openai_api_version,
        )
    else:
        embedding_model = EmbeddingModel(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    return IngestionServices(
        loader_factory=DocumentLoaderFactory(),
        chunker=DocumentChunker(),
        embedding_model=embedding_model,
        qdrant_store=QdrantStore.from_settings(),
    )
