"""Single place to construct chat models, so provider/model choices stay in config."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from etic_genai.config import Settings, get_settings
from etic_genai.core.exceptions import ConfigurationError


def _apply_provider_credentials(
    provider: str | None,
    settings: Settings,
    kwargs: dict[str, Any],
) -> None:
    """Validate and pass credentials for providers configured by this template."""
    if provider is None:
        return

    normalized_provider = provider.lower()

    if normalized_provider == "openai":
        api_key = kwargs.get("api_key") or settings.openai_api_key
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        kwargs.setdefault("api_key", api_key)
        return

    if normalized_provider == "anthropic":
        api_key = kwargs.get("api_key") or settings.anthropic_api_key
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        kwargs.setdefault("api_key", api_key)
        return

    if normalized_provider == "azure_openai":
        api_key = kwargs.get("api_key") or settings.azure_openai_api_key
        endpoint = kwargs.get("azure_endpoint") or settings.azure_openai_endpoint
        api_version = kwargs.get("api_version") or settings.openai_api_version
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "OPENAI_API_VERSION": api_version,
            }.items()
            if not value
        ]
        if missing:
            raise ConfigurationError(
                f"{', '.join(missing)} required when LLM_PROVIDER=azure_openai."
            )
        kwargs.setdefault("api_key", api_key)
        kwargs.setdefault("azure_endpoint", endpoint)
        kwargs.setdefault("api_version", api_version)


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
    _apply_provider_credentials(provider, settings, kwargs)
    return init_chat_model(model, model_provider=provider, **kwargs)
