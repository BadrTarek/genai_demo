from pathlib import Path

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.processing.metadata import fallback_title


class DocxDocumentLoader(DocumentLoader):
    """Converts Word headings to Markdown headings so the chunker can split on sections."""

    supported_extensions = {".docx"}

    def _read(self, path: Path) -> tuple[str, str]:
        import docx

        word_document = docx.Document(str(path))
        lines: list[str] = []
        for paragraph in word_document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            if style.startswith("Heading") and style[-1:].isdigit():
                lines.append(f"{'#' * min(int(style[-1]), 3)} {text}")
            else:
                lines.append(text)
        title = (word_document.core_properties.title or "").strip() or fallback_title(path)
        return "\n\n".join(lines), title
