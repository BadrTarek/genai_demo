"""Application settings, loaded from environment variables and `.env`."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    app_env: str = "dev"
    log_level: str = "INFO"

    # LLM defaults used by `etic_genai.core.llm.get_llm`.
    llm_provider: str | None = "openai"
    llm_model: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_MODEL", "DEPLOYMENT_NAME")
    )
    # None = provider default (some reasoning models reject a temperature).
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None

    # Knowledge base sources
    knowledge_base_path: str = Field(
        default="./requirements",
        validation_alias=AliasChoices("KNOWLEDGE_BASE_PATH", "REQUIREMENTS_DIR"),
    )
    # Directories outside this root are rejected. Defaults to `knowledge_base_path`.
    knowledge_base_root: str | None = None
    index_output_path: str | None = "output/chunks.json"
    max_files_per_index: int = 1000
    max_file_size_mb: float = 20.0
    indexing_on_startup: bool = True

    # Vector store
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = "./data/qdrant"
    qdrant_collection_name: str = "business_requirements"

    # Embeddings (OpenAI-compatible endpoint, configured via OPENAI_API_KEY / OPENAI_BASE_URL)
    embedding_model: str = "text-embedding-3-small"
    embedding_deployment_name: str | None = None

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    max_search_results: int = 5
    min_relevance_score: float = 0.3

    # Guardrails
    max_question_chars: int = 2000
    groundedness_threshold: float = 0.7
    max_generation_attempts: int = 2

    # Evaluation
    online_evaluation: bool = True

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_deployment_name or self.embedding_model


@lru_cache
def get_settings() -> Settings:
    # Export `.env` into os.environ so provider SDKs pick up their API keys.
    load_dotenv()
    return Settings()
