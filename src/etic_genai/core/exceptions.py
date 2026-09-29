"""Project-wide exception hierarchy. Catch `EticGenAIError` to handle all of them."""


class EticGenAIError(Exception):
    """Base class for all project errors."""


class ConfigurationError(EticGenAIError):
    """Missing or invalid configuration."""


class PromptNotFoundError(EticGenAIError):
    """A prompt template file could not be found."""


class ToolNotFoundError(EticGenAIError):
    """A tool name is not present in the registry."""


class AgentError(EticGenAIError):
    """An agent failed to build or run."""
