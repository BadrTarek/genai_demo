from etic_genai.tools.document_access import sqllite_store


def test_user_identity_can_be_saved_and_loaded(tmp_path, monkeypatch):
    db_path = tmp_path / "nested" / "microsoft_auth.db"
    monkeypatch.setattr(sqllite_store, "_db_path", lambda: db_path)

    sqllite_store.upsert_user_identity(
        "user@example.com",
        account="user@example.com",
        tenant_id="tenant-id",
    )

    assert sqllite_store.get_user_identity("user@example.com") == {
        "account": "user@example.com",
        "tenant_id": "tenant-id",
    }
