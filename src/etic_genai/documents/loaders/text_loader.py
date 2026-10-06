from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.models.document import Document


class TextDocumentLoader(DocumentLoader):
    supported_extensions = {".md", ".txt", ".csv"}

    def load(self, path: Path, root: Path) -> Document:
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        if resolved_root not in resolved_path.parents:
            raise ValueError("Document path is outside the configured root")

        content = resolved_path.read_text(encoding="utf-8")
        relative_path = resolved_path.relative_to(resolved_root)

        return Document(
            document_id=relative_path.as_posix(),
            path=resolved_path,
            title=resolved_path.stem.replace("_", " ").replace("-", " ").title(),
            content=content,
            file_type=resolved_path.suffix.lower().removeprefix("."),
        )