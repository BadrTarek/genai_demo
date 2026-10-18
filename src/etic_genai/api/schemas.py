"""Request and response models for the HTTP API."""

import re

from pydantic import BaseModel, Field, field_validator


class ChunkResponse(BaseModel):
    chunk_id: str
    section: str | None
    content: str


class UploadResponse(BaseModel):
    document_id: str
    document_title: str
    chunk_count: int
    chunks: list[ChunkResponse]


class ChatRequest(BaseModel):
    """Message sent to the assistant; reuse conversation_id to continue a chat."""

    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Reuse a prior ID to continue that conversation. Omit to start a new one.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Summarize the uploaded requirements.",
                    "conversation_id": "demo-conversation",
                }
            ]
        }
    }


class RetrievedSource(BaseModel):
    document_title: str
    source: str
    section: str | None
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    sources: list[RetrievedSource]


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("A valid email address is required.")
        return value

    @field_validator("password")
    @classmethod
    def validate_password_policy(cls, value: str) -> str:
        requirements = (
            any(char.isupper() for char in value),
            any(char.islower() for char in value),
            any(char.isdigit() for char in value),
            any(not char.isalnum() for char in value),
        )
        if not all(requirements):
            raise ValueError(
                "Password must include uppercase, lowercase, number, and special characters."
            )
        return value


class SignInRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class UserResponse(BaseModel):
    user_id: str
    email: str


class SignInResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ConversationUsageResponse(BaseModel):
    user_id: str
    email: str
    conversation_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_calls: int
    unreported_calls: int
    updated_at: int
