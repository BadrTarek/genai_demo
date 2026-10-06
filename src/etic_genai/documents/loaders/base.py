from abc import ABC, abstractmethod
from pathlib import Path

from etic_genai.documents.models.document import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path, root: Path) -> Document:
        """Load one document from disk."""