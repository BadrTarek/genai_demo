import json
from pathlib import Path


def run_chunking() -> dict[str, list[dict]]:
    from etic_genai.config import get_settings
    from etic_genai.documents.indexing.qdrant_store import QdrantStore
    from etic_genai.documents.loaders.factory import DocumentLoaderFactory
    from etic_genai.documents.processing.chunker import DocumentChunker
    from etic_genai.embeddings.model import EmbeddingModel

    settings = get_settings()
    root = Path(settings.requirements_dir or "./requirements").resolve()
    output_path = Path("output/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    factory = DocumentLoaderFactory()
    chunker = DocumentChunker()
    output: dict[str, list[dict]] = {}
    if settings.azure_openai_endpoint:
        embedding_model = EmbeddingModel(
            model=settings.embedding_model,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.embedding_deployment_name,
            api_version=settings.openai_api_version,
        )
    else:
        embedding_model = EmbeddingModel(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    qdrant_store = QdrantStore.from_settings()

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            loader = factory.get_loader(path=file_path)
        except ValueError:
            print(f"Skipping unsupported file: {file_path}")
            continue

        document = loader.load(path=file_path, root=root)
        chunks = chunker.split(document)

        # embed the chunks and store them in the output dictionary
        vectors = embedding_model.embed_documents([chunk.content for chunk in chunks])
        qdrant_store.upsert(chunks, vectors)

        # searching
        query_query = embedding_model.embed_query("What are the requirements for the new feature?")
        search_results = qdrant_store.search(query_query, limit=5)
        for result in search_results:
            print(result.score)
            print(result.payload["document_title"])
            print(result.payload["section"])
            print(result.payload["content"])

        output[document.document_id] = [
            {
                "chunk_id": f"{document.document_id}:{chunk.chunk_index}",
                "document_id": document.document_id,
                "document_title": document.title,
                "source": chunk.source,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            for chunk in chunks
        ]

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {sum(len(chunks) for chunks in output.values())} chunks to {output_path}")
    return output


if __name__ == "__main__":
    run_chunking()
