from typing import Any

from etic_genai.prompts.base import BasePrompt


class RAGAnswerPrompt(BasePrompt):
    """System prompt for grounded, cited answers. Rendered with `str.format` because the
    numbered passages are inserted per request."""

    template_name = "rag_answer"

    def render(self, **values: Any) -> str:
        return self.system_template.format(**values)
