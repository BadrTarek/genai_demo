"""Base class for prompts.

Prompt text lives in `prompts/templates/<name>.md` so it can be reviewed and edited
without touching code. A prompt class gives that text a name and turns it into a
`ChatPromptTemplate`.
"""

from abc import ABC
from pathlib import Path
from typing import Any, ClassVar

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from etic_genai.core.exceptions import PromptNotFoundError

TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    path = TEMPLATES_DIR / f"{name}.md"
    if not path.is_file():
        raise PromptNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


class BasePrompt(ABC):
    """A named system prompt followed by the conversation history.

    Subclasses set `template_name` (a file in `templates/`) or override
    `system_template` to build the text in code.
    """

    template_name: ClassVar[str]
    messages_key: ClassVar[str] = "messages"

    @property
    def system_template(self) -> str:
        return load_template(self.template_name)

    def to_chat_prompt(self, **partials: Any) -> ChatPromptTemplate:
        """Build the prompt. `partials` pre-fill template variables such as `{agent_name}`."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                MessagesPlaceholder(self.messages_key),
            ]
        )
        return prompt.partial(**partials) if partials else prompt
