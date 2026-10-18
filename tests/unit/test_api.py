from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import etic_genai.api.dependencies as api_dependencies
import etic_genai.api.routes.evaluations as evaluations_route
from etic_genai.api.dependencies import (
    get_agent,
    get_auth_store,
    get_current_user,
    get_ingestion_services,
    get_usage_store,
)
from etic_genai.api.main import app
from etic_genai.auth.store import AuthStore, AuthUser
from etic_genai.documents.models.document import Document
from etic_genai.documents.models.DocumentChunk import DocumentChunk
from etic_genai.usage.store import ConversationUsage


class FakeLoader:
    def load(self, path: Path, _root: Path) -> Document:
        return Document(
            document_id=path.name,
            path=path,
            title="Test document",
            content="Test content",
            file_type=path.suffix.removeprefix("."),
            metadata={"source": path.name},
        )


class FakeLoaderFactory:
    def get_loader(self, path: Path) -> FakeLoader:
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".pdf"}:
            raise ValueError("unsupported")
        return FakeLoader()


def setup_function() -> None:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        user_id="test-user",
        email="test@example.com",
    )
    app.dependency_overrides[get_usage_store] = lambda: SimpleNamespace(
        add_usage=lambda *_args: None,
        list_all=lambda: [],
    )


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_upload_chunks_embeds_and_stores_document():
    chunk = DocumentChunk(
        document_id="example.md",
        document_title="Test document",
        source="example.md",
        chunk_index=0,
        section="Intro",
        content="Test content",
    )
    stored: dict[str, object] = {}
    services = SimpleNamespace(
        loader_factory=FakeLoaderFactory(),
        chunker=SimpleNamespace(split=lambda _document: [chunk]),
        embedding_model=SimpleNamespace(embed_documents=lambda _texts: [[0.0] * 1536]),
        qdrant_store=SimpleNamespace(
            upsert=lambda chunks, vectors, owner_id, is_public=False: stored.update(
                chunks=chunks,
                vectors=vectors,
                owner_id=owner_id,
                is_public=is_public,
            )
        ),
    )
    app.dependency_overrides[get_ingestion_services] = lambda: services

    response = TestClient(app).post(
        "/documents/upload",
        files={"file": ("example.md", b"# Test document\nTest content", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 1
    assert response.json()["chunks"][0]["content"] == "Test content"
    assert stored["chunks"] == [chunk]
    assert stored["owner_id"] == "test-user"
    assert stored["is_public"] is False

    public_response = TestClient(app).post(
        "/documents/upload",
        data={"visibility": "public"},
        files={"file": ("shared.md", b"# Shared document\nShared content", "text/markdown")},
    )

    assert public_response.status_code == 200
    assert stored["owner_id"] == "test-user"
    assert stored["is_public"] is True


def test_upload_rejects_unsupported_document_type():
    app.dependency_overrides[get_ingestion_services] = lambda: SimpleNamespace(
        loader_factory=FakeLoaderFactory()
    )

    response = TestClient(app).post(
        "/documents/upload",
        files={"file": ("example.exe", b"data", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_chat_reuses_conversation_id():
    calls: list[tuple[str, str]] = []
    usage_updates: list[tuple[str, str, int, int]] = []

    class FakeAgent:
        def invoke(self, input: dict, thread_id: str, callbacks: list) -> dict:
            calls.append((input, thread_id))
            callback = callbacks[0]
            callback.on_llm_end(
                SimpleNamespace(
                    generations=[
                        [
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    usage_metadata={
                                        "input_tokens": 10,
                                        "output_tokens": 4,
                                        "total_tokens": 14,
                                    }
                                )
                            )
                        ]
                    ],
                    llm_output=None,
                )
            )
            return {
                "messages": [SimpleNamespace(content=f"reply to {input['messages'][-1].content}")]
            }

    class FakeUsageStore:
        def add_usage(self, user_id, conversation_id, tracker):
            usage_updates.append(
                (user_id, conversation_id, tracker.input_tokens, tracker.output_tokens)
            )

    services = SimpleNamespace(
        embedding_model=SimpleNamespace(embed_query=lambda query: [0.1] * 1536),
        qdrant_store=SimpleNamespace(
            search=lambda vector, query_text, limit, owner_id=None, public_only=False: (
                [
                    SimpleNamespace(
                        payload={
                            "document_title": "Authentication",
                            "source": "authentication.md",
                            "section": "Lockout",
                            "content": "Accounts lock after five failed attempts.",
                        },
                        score=0.91,
                    )
                ]
                if owner_id == "test-user" and not public_only
                else []
            )
        ),
    )
    app.dependency_overrides[get_agent] = lambda: FakeAgent()
    app.dependency_overrides[get_ingestion_services] = lambda: services
    app.dependency_overrides[get_usage_store] = lambda: FakeUsageStore()
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"message": "first", "conversation_id": "conversation-1"},
    )
    second = client.post(
        "/chat",
        json={"message": "second", "conversation_id": "conversation-1"},
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["conversation_id"] == "conversation-1"
    assert calls[1][0]["messages"][-1].content == "second"
    assert "Accounts lock after five failed attempts." in calls[0][0]["retrieved_context"]
    assert calls[0][0]["messages"][-1].content == "first"
    assert calls[0][1] == calls[1][1] == "test-user:conversation-1"
    assert first.json()["sources"][0]["source"] == "authentication.md"
    assert usage_updates == [
        ("test-user", "conversation-1", 10, 4),
        ("test-user", "conversation-1", 10, 4),
    ]


def test_register_and_sign_in_with_sqlite(tmp_path: Path):
    auth_store = AuthStore(str(tmp_path / "auth.db"))
    app.dependency_overrides[get_auth_store] = lambda: auth_store
    client = TestClient(app)

    registration = client.post(
        "/auth/register",
        json={"email": "Person@Example.com", "password": "StrongPassword1!"},
    )
    duplicate = client.post(
        "/auth/register",
        json={"email": "person@example.COM", "password": "AnotherPassword2!"},
    )
    sign_in = client.post(
        "/auth/sign-in",
        json={"email": "PERSON@example.com", "password": "StrongPassword1!"},
    )

    assert registration.status_code == 201
    assert registration.json()["email"] == "person@example.com"
    assert duplicate.status_code == 409
    assert sign_in.status_code == 200
    token = sign_in.json()["access_token"]
    assert auth_store.get_active_session_user(token).email == "person@example.com"


def test_protected_endpoints_require_authentication():
    app.dependency_overrides.pop(get_current_user)
    app.dependency_overrides[get_agent] = lambda: SimpleNamespace()
    app.dependency_overrides[get_ingestion_services] = lambda: SimpleNamespace(
        embedding_model=SimpleNamespace(embed_query=lambda _query: [0.0] * 1536),
        qdrant_store=SimpleNamespace(search=lambda _vector, query_text, limit: []),
    )

    response = TestClient(app).post(
        "/chat",
        json={"message": "hello"},
    )

    assert response.status_code == 401
    assert "required" in response.json()["detail"].lower()


def test_swagger_ui_and_openapi_documentation_are_available():
    client = TestClient(app)

    docs_response = client.get("/docs")
    schema_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert "swagger-ui" in docs_response.text.lower()
    assert schema_response.status_code == 200

    schema = schema_response.json()
    assert schema["info"]["title"] == "ETIC GenAI API"
    assert schema["paths"]["/auth/register"]["post"]["tags"] == ["Authentication"]
    assert schema["paths"]["/auth/sign-in"]["post"]["tags"] == ["Authentication"]
    assert schema["paths"]["/documents/upload"]["post"]["tags"] == ["Documents"]
    assert schema["paths"]["/chat"]["post"]["tags"] == ["Chat"]
    assert schema["paths"]["/admin/usage"]["get"]["tags"] == ["Usage"]
    assert schema["paths"]["/admin/evaluations/run"]["post"]["tags"] == ["Evaluations"]


def test_admin_evaluation_endpoint_requires_admin_access():
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        user_id="regular-user",
        email="regular@example.com",
    )

    response = TestClient(app).post("/admin/evaluations/run")

    assert response.status_code == 403


def test_admin_evaluation_endpoint_returns_experiment_details(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        user_id="admin-user",
        email="admin@example.com",
        is_admin=True,
    )
    requested_user_ids: list[str] = []

    def fake_run_evaluation(user_id: str) -> dict[str, str | None]:
        requested_user_ids.append(user_id)
        return {
            "status": "completed",
            "experiment_name": "brd-rag-test",
            "experiment_id": "experiment-id",
            "experiment_url": "https://smith.langchain.com/experiment",
        }

    monkeypatch.setattr(evaluations_route, "run_dataset_evaluation", fake_run_evaluation)

    response = TestClient(app).post("/admin/evaluations/run")

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "experiment_name": "brd-rag-test",
        "experiment_id": "experiment-id",
        "experiment_url": "https://smith.langchain.com/experiment",
    }
    assert requested_user_ids == ["admin-user"]


def test_admin_usage_endpoint_requires_admin_access():
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        user_id="regular-user",
        email="regular@example.com",
    )

    response = TestClient(app).get("/admin/usage")

    assert response.status_code == 403


def test_admin_usage_endpoint_lists_usage_for_all_conversations():
    expected_usage = ConversationUsage(
        user_id="user-1",
        email="user@example.com",
        conversation_id="conversation-1",
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        model_calls=2,
        unreported_calls=0,
        updated_at=1_700_000_000,
    )
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        user_id="admin-user",
        email="admin@example.com",
        is_admin=True,
    )
    app.dependency_overrides[get_usage_store] = lambda: SimpleNamespace(
        list_all=lambda: [expected_usage],
    )

    response = TestClient(app).get("/admin/usage")

    assert response.status_code == 200
    assert response.json() == [
        {
            "user_id": "user-1",
            "email": "user@example.com",
            "conversation_id": "conversation-1",
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "model_calls": 2,
            "unreported_calls": 0,
            "updated_at": 1_700_000_000,
        }
    ]


def test_ingestion_services_use_configured_azure_embedding_settings(monkeypatch):
    captured: dict[str, object] = {}
    settings = SimpleNamespace(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="azure-key",
        embedding_deployment_name="embedding-deployment",
        openai_api_version="2024-10-21",
        embedding_model="configured-embedding-model",
        openai_api_key=None,
        openai_base_url=None,
    )

    class FakeEmbeddingModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(api_dependencies, "EmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(
        api_dependencies.QdrantStore,
        "from_settings",
        classmethod(lambda cls: SimpleNamespace()),
    )
    api_dependencies.get_ingestion_services.cache_clear()
    try:
        services = api_dependencies.get_ingestion_services()
    finally:
        api_dependencies.get_ingestion_services.cache_clear()

    assert isinstance(services, api_dependencies.IngestionServices)
    assert captured == {
        "model": "configured-embedding-model",
        "api_key": "azure-key",
        "azure_endpoint": "https://example.openai.azure.com",
        "azure_deployment": "embedding-deployment",
        "api_version": "2024-10-21",
    }


def test_ingestion_services_use_configured_openai_embedding_settings(monkeypatch):
    captured: dict[str, object] = {}
    settings = SimpleNamespace(
        azure_openai_endpoint=None,
        azure_openai_api_key=None,
        embedding_deployment_name=None,
        openai_api_version=None,
        embedding_model="configured-embedding-model",
        openai_api_key="openai-key",
        openai_base_url="https://api.example.com/v1",
    )

    class FakeEmbeddingModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(api_dependencies, "EmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(
        api_dependencies.QdrantStore,
        "from_settings",
        classmethod(lambda cls: SimpleNamespace()),
    )
    api_dependencies.get_ingestion_services.cache_clear()
    try:
        services = api_dependencies.get_ingestion_services()
    finally:
        api_dependencies.get_ingestion_services.cache_clear()

    assert isinstance(services, api_dependencies.IngestionServices)
    assert captured == {
        "model": "configured-embedding-model",
        "api_key": "openai-key",
        "base_url": "https://api.example.com/v1",
    }
