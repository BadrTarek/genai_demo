from pathlib import Path

import pytest
from pypdf import PdfWriter

from etic_genai.documents.loaders import DocumentLoaderFactory
from etic_genai.documents.processing.chunker import DocumentChunker


def test_index_then_search_finds_relevant_source(knowledge_base, requirements_dir: Path):
    report = knowledge_base.index_path(str(requirements_dir))

    assert report["documents_indexed"] == 3
    assert report["failed"] == []
    hits = knowledge_base.search("How many characters must passwords contain?", top_k=1)
    assert hits[0]["source"] == "authentication.md"
    assert hits[0]["section"] == "Passwords"


def test_reindex_is_incremental(knowledge_base, requirements_dir: Path):
    knowledge_base.index_path(str(requirements_dir))

    report = knowledge_base.index_path(str(requirements_dir))
    assert report["documents_indexed"] == 0
    assert report["documents_unchanged"] == 3

    (requirements_dir / "invoice_approval.txt").write_text("Invoices need two approvers.")
    report = knowledge_base.index_path(str(requirements_dir))
    assert report["documents_indexed"] == 1
    assert report["documents_unchanged"] == 2


def test_deleted_and_shrunk_documents_leave_no_stale_chunks(knowledge_base, requirements_dir):
    knowledge_base.index_path(str(requirements_dir))
    (requirements_dir / "user_roles.csv").unlink()
    (requirements_dir / "authentication.md").write_text("# Authentication\nShort now.")

    report = knowledge_base.index_path(str(requirements_dir))

    assert report["documents_removed"] == ["user_roles.csv"]
    documents = {doc["document_id"]: doc["chunks"] for doc in knowledge_base.list_documents()}
    assert documents == {"authentication.md": 1, "invoice_approval.txt": 1}


def test_one_broken_file_does_not_abort_indexing(knowledge_base, requirements_dir: Path):
    (requirements_dir / "broken.txt").write_bytes(b"\xff\xfe\x00 not utf-8 \x81")

    report = knowledge_base.index_path(str(requirements_dir))

    assert report["documents_indexed"] == 3
    assert [f["document_id"] for f in report["failed"]] == ["broken.txt"]


def test_rejects_path_outside_allowed_root(knowledge_base, tmp_path: Path):
    with pytest.raises(ValueError, match="outside the allowed root"):
        knowledge_base.index_path(str(tmp_path.parent))


def test_search_on_empty_store_returns_nothing(knowledge_base):
    assert knowledge_base.search("anything") == []


def test_csv_rows_become_self_describing_lines(requirements_dir: Path):
    path = requirements_dir / "user_roles.csv"
    document = DocumentLoaderFactory().get_loader(path).load(path, requirements_dir)

    assert document.content == (
        "role: finance_manager; permission: invoice.reopen; description: Reopen approved invoices"
    )


def test_pdf_without_text_or_title_loads(tmp_path: Path):
    path = tmp_path / "docs" / "empty_policy.pdf"
    path.parent.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as file:
        writer.write(file)

    document = DocumentLoaderFactory().get_loader(path).load(path, path.parent)

    assert document.title == "Empty Policy"
    assert document.content == ""
    assert DocumentChunker().split(document) == []


def test_markdown_chunks_carry_section_and_title_context(requirements_dir: Path):
    path = requirements_dir / "authentication.md"
    document = DocumentLoaderFactory().get_loader(path).load(path, requirements_dir)

    chunk = DocumentChunker().split(document)[0]

    assert chunk.section == "Passwords"
    assert chunk.embedding_text.startswith("Document: Authentication\nSection: Passwords")


def test_reset_of_local_store_removes_all_points(tmp_path: Path, requirements_dir: Path):
    from etic_genai.documents.indexing.knowledge_base import KnowledgeBase
    from etic_genai.documents.indexing.qdrant_store import QdrantStore
    from tests.conftest import BagOfWordsEmbeddings

    def open_kb() -> KnowledgeBase:
        return KnowledgeBase(QdrantStore(path=str(tmp_path / "qdrant")), BagOfWordsEmbeddings())

    knowledge_base = open_kb()
    knowledge_base.index_path(str(requirements_dir))
    knowledge_base.reset()
    knowledge_base.store.client.close()

    assert open_kb().list_documents() == []
