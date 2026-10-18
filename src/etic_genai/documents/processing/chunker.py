from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from etic_genai.core.logging import get_logger
from etic_genai.documents.models import Document, DocumentChunk

logger = get_logger(__name__)

# File types whose content uses Markdown headings (docx headings are converted on load).
MARKDOWN_FILE_TYPES = {"md", "docx"}


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: Document) -> list[DocumentChunk]:
        if document.file_type in MARKDOWN_FILE_TYPES:
            sections = self.markdown_splitter.split_text(document.content)
        else:
            sections = [LangChainDocument(page_content=document.content)]

        split_documents = [
            piece
            for piece in self.text_splitter.split_documents(sections)
            if piece.page_content.strip()
        ]
        logger.debug("Split %s into %d chunks", document.document_id, len(split_documents))

        return [
            DocumentChunk(
                document_id=document.document_id,
                document_title=document.title,
                source=document.metadata.get("source", document.document_id),
                chunk_index=index,
                section=self._section(piece.metadata),
                content=piece.page_content,
                document_hash=document.content_hash or "",
                file_type=document.file_type,
            )
            for index, piece in enumerate(split_documents)
        ]

    @staticmethod
    def _section(metadata: dict) -> str | None:
        # h1 is usually the document title, so the section is the h2/h3 path.
        parts = [metadata[key] for key in ("h2", "h3") if metadata.get(key)]
        return " > ".join(parts) or metadata.get("h1")
