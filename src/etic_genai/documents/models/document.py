from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    path: Path
    title: str
    content: str
    file_type: str
    metadata: dict[str, str] = field(default_factory=dict)
    content_hash: str | None = ""
    modified_at: datetime | None = None
