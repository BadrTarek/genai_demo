"""Single place to construct chat models, so provider/model choices stay in config."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from etic_genai.config import get_settings
from etic_genai.core.exceptions import ConfigurationError


def get_llm(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model. Arguments override the `LLM_*` settings from `.env`."""
    settings = get_settings()
    model = model or settings.llm_model
    provider = provider or settings.llm_provider
    if not model:
        raise ConfigurationError("No LLM model configured. Set LLM_MODEL in .env.")

    kwargs.setdefault("temperature", settings.llm_temperature)
    if settings.llm_max_tokens is not None:
        kwargs.setdefault("max_tokens", settings.llm_max_tokens)
    return init_chat_model(model, model_provider=provider, **kwargs)
