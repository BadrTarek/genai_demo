import pytest

from etic_genai.core.exceptions import ConfigurationError
from etic_genai.core.llm import get_llm


def test_get_llm_passes_openai_api_key_from_settings(monkeypatch):
    captured = {}

    def fake_init_chat_model(model, model_provider=None, **kwargs):
        captured["model"] = model
        captured["model_provider"] = model_provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("etic_genai.core.llm.init_chat_model", fake_init_chat_model)

    from etic_genai.config.settings import get_settings

    get_settings.cache_clear()
    get_llm()

    assert captured["model"] == "gpt-4o-mini"
    assert captured["model_provider"] == "openai"
    assert captured["kwargs"]["api_key"] == "test-key"


def test_get_llm_requires_openai_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    from etic_genai.config.settings import get_settings

    get_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        get_llm()
