from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.processing.metadata import extract_markdown_title, fallback_title


class TextDocumentLoader(DocumentLoader):
    supported_extensions = {".md", ".txt"}

    def _read(self, path: Path) -> tuple[str, str]:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".md":
            return content, extract_markdown_title(content, path)
        return content, fallback_title(path)
