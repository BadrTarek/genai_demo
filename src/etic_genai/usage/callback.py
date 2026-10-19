"""Collect token usage reported by LangChain chat model calls."""

from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageTracker(BaseCallbackHandler):
    """Accumulate token usage and model-call counts for one agent invocation."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.model_calls = 0
        self.unreported_calls = 0
        self._lock = Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        del kwargs
        usage = self._get_usage(response)
        with self._lock:
            self.model_calls += 1
            if usage is None:
                self.unreported_calls += 1
                return
            input_tokens, output_tokens, total_tokens = usage
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens

    @staticmethod
    def _get_usage(response: LLMResult) -> tuple[int, int, int] | None:
        for generation_group in response.generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None)
                if isinstance(usage_metadata, dict):
                    return UsageTracker._usage_tuple(usage_metadata)

        if isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage")
            if isinstance(token_usage, dict):
                return UsageTracker._usage_tuple(token_usage)
        return None

    @staticmethod
    def _usage_tuple(usage: dict[str, Any]) -> tuple[int, int, int]:
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        return input_tokens, output_tokens, total_tokens
