"""Embedding model factory. Everything else depends on LangChain's `Embeddings` interface."""

import os

from langchain_core.embeddings import Embeddings

from etic_genai.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
