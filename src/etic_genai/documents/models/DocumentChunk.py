from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    document_title: str
    source: str
    chunk_index: int = 0
    section: str | None = None
    content: str = ""
    document_hash: str = ""
    file_type: str = ""

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"

    @property
    def embedding_text(self) -> str:
        """Text that gets embedded: the chunk plus its title/section, so short chunks
        like "must contain at least 12 characters" keep their context."""
        header = f"Document: {self.document_title}"
        if self.section:
            header += f"\nSection: {self.section}"
        return f"{header}\n\n{self.content}"
