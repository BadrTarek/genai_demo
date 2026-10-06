from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    path: Path
    title: str
    content: str
    file_type: str