from collections.abc import Sequence

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


class EmbeddingModel:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
    ) -> None:
        if azure_endpoint is not None:
            if azure_deployment is None or api_version is None:
                raise ValueError(
                    "Azure embeddings require an Azure deployment name and API version."
                )
            self.model = AzureOpenAIEmbeddings(
                model=model,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                api_version=api_version,
                api_key=api_key,
            )
        else:
            self.model = OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)