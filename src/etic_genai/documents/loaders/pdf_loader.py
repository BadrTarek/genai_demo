from pathlib import Path

from pypdf import PdfReader

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.processing.metadata import fallback_title


class PdfDocumentLoader(DocumentLoader):
    supported_extensions = {".pdf"}

    def _read(self, path: Path) -> tuple[str, str]:
        reader = PdfReader(path)
        content = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
        pdf_title = reader.metadata.title if reader.metadata else None
        return content, (pdf_title or "").strip() or fallback_title(path)
