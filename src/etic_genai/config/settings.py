"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
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
    # document indexing
    requirements_dir: str | None = "./requirements"
    sqlite_db_path: str | None = "./data/documents.db"
    qdrant_mode: Literal["local", "docker"] = "local"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str | None = "./data/qdrant"
    qdrant_collection_name: str | None = "business_requirements"
    # Embeddings
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDINGS_MODEL",
    )
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    openai_api_version: str | None = None
    embedding_deployment_name: str | None = None
    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Indexing
    indexing_on_startup: bool = True
    max_search_results: int = 5

    allowed_file_types: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt", "md", "csv", "json"]
    )


@lru_cache
def get_settings() -> Settings:
    # Export `.env` into os.environ so provider SDKs pick up their API keys.
    load_dotenv()
    return Settings()
