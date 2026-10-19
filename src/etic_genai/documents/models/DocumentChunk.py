
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    document_title: str
    source: str
    chunk_index: int = 0
    section: str | None = None
    content: str = ""
    
    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"
    