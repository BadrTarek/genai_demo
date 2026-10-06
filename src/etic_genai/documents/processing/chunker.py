from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from etic_genai.documents.models import DocumentChunk, Document
class DocumentChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
            ],
            strip_headers=True,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: Document) -> list[DocumentChunk]:
        if document.file_type == "md":
            print(f" other Document title : {document.title}")
            sections = self.markdown_splitter.split_text(document.content)
        else:
            print(f"Document title : {document.title}")
            sections = [
                LangChainDocument(page_content=document.content, metadata=document.metadata)
            ]

        split_documents = self.text_splitter.split_documents(sections)

        return [
            DocumentChunk(
                document_id=document.document_id,
                document_title=document.title,
                source=document.metadata.get("source", document.document_id),
                chunk_index=index,
                section=split_document.metadata.get("section"),
                content=split_document.page_content,
            )
            for index, split_document in enumerate(split_documents)
        ]