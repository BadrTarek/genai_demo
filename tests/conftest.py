from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeToolChatModel(GenericFakeChatModel):
    """Fake chat model that accepts `bind_tools`, so agents can be tested offline."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolChatModel":
        return self


@pytest.fixture
def fake_llm_factory():
    def make(*replies: AIMessage | str) -> FakeToolChatModel:
        messages = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
        return FakeToolChatModel(messages=iter(messages))

    return make
