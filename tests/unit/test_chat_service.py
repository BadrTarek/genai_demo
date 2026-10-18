from types import SimpleNamespace

from etic_genai.api.services.chat import run_chat_agent
from etic_genai.auth.store import AuthUser


def _match(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        payload={
            "document_title": "Reference",
            "source": "reference.md",
            "section": "Overview",
            "content": content,
        },
        score=0.9,
    )


def test_chat_uses_owned_documents_without_public_fallback():
    searches: list[dict[str, object]] = []
    private_match = _match("private content")

    class FakeQdrant:
        def search(self, vector, query_text, limit, owner_id=None, public_only=False):
            searches.append(
                {"query_text": query_text, "owner_id": owner_id, "public_only": public_only}
            )
            return [private_match] if owner_id == "user-1" else []

    class FakeAgent:
        def invoke(self, input, thread_id, callbacks=None):
            return {"messages": [SimpleNamespace(content=input["retrieved_context"])]}

    result = run_chat_agent(
        message="question",
        conversation_id="chat-1",
        current_user=AuthUser(user_id="user-1", email="user@example.com"),
        agent=FakeAgent(),
        services=SimpleNamespace(
            embedding_model=SimpleNamespace(embed_query=lambda _message: [0.1]),
            qdrant_store=FakeQdrant(),
        ),
    )

    assert searches == [{"query_text": "question", "owner_id": "user-1", "public_only": False}]
    assert "private content" in result.answer
    assert "private content" in result.context
    assert result.sources[0].source == "reference.md"


def test_chat_falls_back_to_public_documents_if_owned_docs_not_found():
    searches: list[dict[str, object]] = []
    public_match = _match("public content")

    class FakeQdrant:
        def search(self, vector, query_text, limit, owner_id=None, public_only=False):
            searches.append(
                {"query_text": query_text, "owner_id": owner_id, "public_only": public_only}
            )
            return [public_match] if public_only else []

    class FakeAgent:
        def invoke(self, input, thread_id, callbacks=None):
            return {"messages": [SimpleNamespace(content=input["retrieved_context"])]}

    result = run_chat_agent(
        message="question",
        conversation_id="chat-1",
        current_user=AuthUser(user_id="user-1", email="user@example.com"),
        agent=FakeAgent(),
        services=SimpleNamespace(
            embedding_model=SimpleNamespace(embed_query=lambda _message: [0.1]),
            qdrant_store=FakeQdrant(),
        ),
    )

    assert searches == [
        {"query_text": "question", "owner_id": "user-1", "public_only": False},
        {"query_text": "question", "owner_id": None, "public_only": True},
    ]
    assert "public content" in result.answer
    assert "public content" in result.context
    assert result.sources[0].source == "reference.md"


def test_chat_does_not_fall_back_when_owned_chunks_exist_without_content():
    searches: list[dict[str, object]] = []

    class FakeQdrant:
        def search(self, vector, query_text, limit, owner_id=None, public_only=False):
            searches.append(public_only)
            return [_match("   ")] if owner_id else [_match("public content")]

    class FakeAgent:
        def invoke(self, input, thread_id, callbacks=None):
            return {"messages": [SimpleNamespace(content=input["retrieved_context"])]}

    result = run_chat_agent(
        message="question",
        conversation_id="chat-1",
        current_user=AuthUser(user_id="user-1", email="user@example.com"),
        agent=FakeAgent(),
        services=SimpleNamespace(
            embedding_model=SimpleNamespace(embed_query=lambda _message: [0.1]),
            qdrant_store=FakeQdrant(),
        ),
    )

    assert searches == [False]
    assert result.sources == []
    assert result.context == "No matching uploaded document excerpts were found."
    assert "No matching uploaded document excerpts were found." in result.answer
