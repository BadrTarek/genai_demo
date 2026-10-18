from pathlib import Path

from pypdf import PdfReader

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.models.document import Document
from etic_genai.documents.processing.metadata import calculate_content_hash, get_modified_at


class PdfDocumentLoader(DocumentLoader):
    supported_extensions = {".pdf"}

    def load(self, path: Path, root: Path) -> Document:
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        if resolved_root not in resolved_path.parents:
            raise ValueError("Document path is outside the configured root")

        reader = PdfReader(resolved_path)
        content = "\n".join(page.extract_text() or "" for page in reader.pages)
        relative_path = resolved_path.relative_to(resolved_root)

        pdf_title = reader.metadata.title if reader.metadata and reader.metadata.title else None
        title = (
            pdf_title
            if pdf_title
            else resolved_path.stem.replace("_", " ").replace("-", " ").title()
        )

        return Document(
            document_id=relative_path.as_posix(),
            path=resolved_path,
            title=title,
            content=content,
            content_hash=calculate_content_hash(content),
            modified_at=get_modified_at(resolved_path),
            file_type="pdf",
            metadata={"source": relative_path.as_posix(), "file_name": resolved_path.name},
        )
