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

    # Provider credentials. Values come from `.env` or the process environment.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    openai_api_version: str | None = None

    # Microsoft Graph delegated token used by SharePoint tools.
    microsoft_graph_access_token: str | None = None

    microsoft_client_id: str = "14d82eec-204b-4c2f-b7e8-296a70dab67e"
    microsoft_tenant_id: str = "common"
    microsoft_auth_db_path: str = ".data/microsoft_auth.db"
    local_files_root: str | None = None

@lru_cache
def get_settings() -> Settings:
    # Export `.env` into os.environ so provider SDKs pick up their API keys.
    load_dotenv()
    return Settings()
