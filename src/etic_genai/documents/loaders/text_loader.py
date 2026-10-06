from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.models.document import Document
from etic_genai.documents.processing.metadata import calculate_content_hash, extract_markdown_title, fallback_title, get_modified_at


class TextDocumentLoader(DocumentLoader):
    supported_extensions = {".md", ".txt", ".csv"}

    def load(self, path: Path, root: Path) -> Document:
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        if resolved_root not in resolved_path.parents:
            raise ValueError("Document path is outside the configured root")

        content = resolved_path.read_text(encoding="utf-8")
        relative_path = resolved_path.relative_to(resolved_root)
        file_type = resolved_path.suffix.lower().removeprefix(".")
        title = (
            extract_markdown_title(content, resolved_path)
            if file_type == "md"
            else fallback_title(resolved_path)
        )

        return Document(
            document_id=relative_path.as_posix(),
            path=resolved_path,
            title=title,
            content=content,
            content_hash=calculate_content_hash(content),
            modified_at=get_modified_at(resolved_path),
            file_type=resolved_path.suffix.lower().removeprefix("."),
            metadata={
                "source": relative_path.as_posix(),
                "file_name": resolved_path.name
            }
        )