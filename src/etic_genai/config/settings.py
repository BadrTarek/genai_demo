"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"

    # LLM defaults used by `etic_genai.core.llm.get_llm`.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int | None = None


@lru_cache
def get_settings() -> Settings:
    # Export `.env` into os.environ so provider SDKs pick up their API keys.
    load_dotenv()
    return Settings()
