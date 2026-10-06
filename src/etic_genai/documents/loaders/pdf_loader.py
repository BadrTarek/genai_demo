from pathlib import Path

from pypdf import PdfReader

from etic_genai.documents.loaders.base import DocumentLoader
from etic_genai.documents.models.document import Document


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

        return Document(
            document_id=relative_path.as_posix(),
            path=resolved_path,
            title=resolved_path.stem.replace("_", " ").replace("-", " ").title(),
            content=content,
            file_type="pdf",
        )