from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from etic_genai.documents.models.document import Document
from etic_genai.documents.processing.metadata import calculate_content_hash, get_modified_at


class DocumentLoader(ABC):
    supported_extensions: ClassVar[set[str]] = set()

    def load(self, path: Path, root: Path) -> Document:
        """Load one document from disk. `path` must be inside `root`."""
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        if resolved_root not in resolved_path.parents:
            raise ValueError("Document path is outside the configured root")

        content, title = self._read(resolved_path)
        relative_path = resolved_path.relative_to(resolved_root).as_posix()
        return Document(
            document_id=relative_path,
            path=resolved_path,
            title=title,
            content=content,
            file_type=resolved_path.suffix.lower().removeprefix("."),
            metadata={"source": relative_path, "file_name": resolved_path.name},
            content_hash=calculate_content_hash(content),
            modified_at=get_modified_at(resolved_path),
        )

    @abstractmethod
    def _read(self, path: Path) -> tuple[str, str]:
        """Return `(content, title)` for the file."""
