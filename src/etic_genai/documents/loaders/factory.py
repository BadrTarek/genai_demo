from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.loaders.csv_loader import CsvDocumentLoader
from etic_genai.documents.loaders.docx_loader import DocxDocumentLoader
from etic_genai.documents.loaders.pdf_loader import PdfDocumentLoader
from etic_genai.documents.loaders.text_loader import TextDocumentLoader


class DocumentLoaderFactory:
    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders = loaders or [
            TextDocumentLoader(),
            CsvDocumentLoader(),
            PdfDocumentLoader(),
            DocxDocumentLoader(),
        ]

    @property
    def supported_extensions(self) -> set[str]:
        return {ext for loader in self._loaders for ext in loader.supported_extensions}

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def get_loader(self, path: Path) -> DocumentLoader:
        suffix = path.suffix.lower()
        for loader in self._loaders:
            if suffix in loader.supported_extensions:
                return loader
        raise ValueError(f"No loader available for file type: {suffix}")
