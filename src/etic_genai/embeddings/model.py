import os
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings


class EmbeddingModel:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = OpenAIEmbeddings(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)