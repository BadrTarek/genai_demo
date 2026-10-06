from pathlib import Path

import pytest

from etic_genai.documents.loaders import DocumentLoaderFactory
from etic_genai.documents.loaders.pdf_loader import PdfDocumentLoader


def test_loads_markdown_document(tmp_path: Path):
    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()

    document_path = requirements_dir / "authentication.md"
    document_path.write_text(
        "# Authentication\nUsers must sign in with email and password.",
        encoding="utf-8",
    )

    loader = DocumentLoaderFactory().get_loader(document_path)
    document = loader.load(document_path, requirements_dir)
    print(document)

    assert document.title == "Authentication"
    assert document.metadata["source"] == "authentication.md"
    assert document.metadata["file_name"] == "authentication.md"
    assert document.content_hash
    assert document.modified_at is not None

def test_extracts_markdown_heading_as_title(tmp_path: Path):
    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()

    document_path = requirements_dir / "security_rules.md"
    document_path.write_text(
        "# Password Security Requirements\nPasswords must be strong.",
        encoding="utf-8",
    )

    loader = DocumentLoaderFactory().get_loader(document_path)
    document = loader.load(document_path, requirements_dir)

    assert document.title == "Password Security Requirements"

def test_rejects_path_outside_root(tmp_path: Path):
    requirements_dir = tmp_path / "requirements"
    requirements_dir.mkdir()

    external_file = tmp_path / "outside.txt"
    external_file.write_text("private content", encoding="utf-8")

    loader = DocumentLoaderFactory().get_loader(external_file)

    with pytest.raises(ValueError, match="outside"):
        loader.load(external_file, requirements_dir)


def test_rejects_unsupported_extension(tmp_path: Path):
    document_path = tmp_path / "requirements.xyz"
    document_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="No loader"):
        DocumentLoaderFactory().get_loader(document_path)
def test_selects_pdf_loader(tmp_path: Path):
    document_path = tmp_path / "requirements.pdf"

    loader = DocumentLoaderFactory().get_loader(document_path)

    assert isinstance(loader, PdfDocumentLoader)