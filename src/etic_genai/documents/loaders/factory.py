from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.loaders.pdf_loader import PdfDocumentLoader
from etic_genai.documents.loaders.text_loader import TextDocumentLoader


class DocumentLoaderFactory:
    def __init__(self) -> None:
        self._loaders: list[DocumentLoader] = [
            TextDocumentLoader(),
            PdfDocumentLoader(),
        ]

    def get_loader(self, path: Path) -> DocumentLoader:
        suffix = path.suffix.lower()

        for loader in self._loaders:
            supported_extensions = getattr(loader, "supported_extensions", set())
            if suffix in supported_extensions:
                return loader

        raise ValueError(f"No loader available for file type: {suffix}")