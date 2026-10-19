from pathlib import Path

import pytest


def test_index_then_hybrid_search_finds_source(indexed_kb):
    hits = indexed_kb.search("How many characters must passwords contain?", top_k=2)

    assert hits[0]["source"] == "authentication.md"
    assert hits[0]["section"] == "Passwords"
    assert 0 < hits[0]["relevance"] <= 1
    assert set(hits[0]) == {
        "score",
        "relevance",
        "document_id",
        "document_title",
        "source",
        "section",
        "content",
    }


def test_documents_are_shared_and_public(indexed_kb):
    payloads = [
        point.payload
        for point in indexed_kb.store.client.scroll(indexed_kb.store.collection_name, limit=100)[0]
    ]
    assert payloads and all(p["is_public"] and p["owner_id"] is None for p in payloads)


def test_reindexing_an_edited_file_leaves_no_stale_chunks(knowledge_base, requirements_dir):
    long_doc = requirements_dir / "long.md"
    long_doc.write_text("\n\n".join(f"## Part {i}\n\n" + "word " * 200 for i in range(4)))
    knowledge_base.index_directory(str(requirements_dir))
    long_doc.write_text("# Long\n\nNow short.")

    knowledge_base.index_directory(str(requirements_dir))

    chunks = {doc["document_id"]: doc["chunks"] for doc in knowledge_base.list_documents()}
    assert chunks["long.md"] == 1


def test_one_broken_file_does_not_abort_indexing(knowledge_base, requirements_dir: Path):
    (requirements_dir / "broken.txt").write_bytes(b"\xff\xfe\x00 not utf-8 \x81")
    (requirements_dir / "notes.xyz").write_text("unsupported")

    report = knowledge_base.index_directory(str(requirements_dir))

    assert report["documents_indexed"] == 3
    assert [f["document_id"] for f in report["failed"]] == ["broken.txt"]
    assert report["skipped"] == ["notes.xyz"]


def test_rejects_path_outside_allowed_root(knowledge_base, tmp_path: Path):
    with pytest.raises(ValueError, match="outside the allowed root"):
        knowledge_base.index_directory(str(tmp_path.parent))


def test_private_user_documents_are_not_searched(knowledge_base, qdrant_store):
    from etic_genai.documents.models.DocumentChunk import DocumentChunk

    chunk = DocumentChunk("secret.md", "Secret", "secret.md", content="private password plan")
    qdrant_store.upsert(
        [chunk], knowledge_base.embedding_model.embed_documents([chunk.content]), owner_id="u1"
    )

    assert knowledge_base.search("private password plan") == []
    assert knowledge_base.list_documents() == []
