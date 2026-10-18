"""Guardrails applied to the user's question before retrieval."""

import re

from etic_genai.guardrails.base import Guardrail, GuardrailContext, GuardrailResult


class InputLengthGuardrail(Guardrail):
    name = "input_length"

    def __init__(self, max_chars: int = 2000, min_chars: int = 3) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        length = len(text.strip())
        if length < self.min_chars:
            return self._fail(text, "The question is empty or too short.")
        if length > self.max_chars:
            return self._fail(
                text, f"The question is too long ({length} > {self.max_chars} characters)."
            )
        return self._pass(text)


INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b"
    r".{0,20}\b(instructions?|prompts?|guidelines|directives)\b",
    r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?)",
    r"\byou are now\b",
    r"\b(jailbreak|developer mode|do anything now)\b",
    r"\bpretend (to be|you are)\b",
    r"<\s*/?\s*(system|assistant)\s*>",
]


class PromptInjectionGuardrail(Guardrail):
    """Heuristic detection of common prompt-injection / jailbreak phrasings.

    Cheap and deterministic. In production you would layer a classifier model
    (or a provider's safety endpoint) behind this; the interface stays the same.
    """

    name = "prompt_injection"

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns or INJECTION_PATTERNS
        ]

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        for pattern in self.patterns:
            if match := pattern.search(text):
                return self._fail(
                    text,
                    "The question looks like a prompt-injection attempt.",
                    match=match.group(0),
                )
        return self._pass(text)


PII_PATTERNS: dict[str, str] = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "IBAN": r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b",
    "CREDIT_CARD": r"\b(?:\d{4}[ -]?){3}\d{4}(?:\d{1,3})?\b",
    "PHONE": r"(?<!\w)\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)",
}


class PIIRedactionGuardrail(Guardrail):
    """Replaces personal data with placeholders such as `[EMAIL]`. Never fails, so it
    can be used both on questions (keep PII away from the LLM) and on answers."""

    name = "pii_redaction"

    def __init__(self, patterns: dict[str, str] | None = None) -> None:
        self.patterns = {
            label: re.compile(pattern) for label, pattern in (patterns or PII_PATTERNS).items()
        }

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        redacted = text
        found: dict[str, int] = {}
        for label, pattern in self.patterns.items():
            redacted, count = pattern.subn(f"[{label}]", redacted)
            if count:
                found[label] = count
        result = self._pass(redacted, redacted=found)
        result.modified = bool(found)
        if found:
            result.reason = "Redacted " + ", ".join(f"{n} {label}" for label, n in found.items())
        return result
