"""FastAPI application assembly."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from etic_genai.api.routes.auth import router as auth_router
from etic_genai.api.routes.chat import router as chat_router
from etic_genai.api.routes.documents import router as documents_router
from etic_genai.api.routes.usage import router as usage_router

API_TAGS = [
    {
        "name": "Authentication",
        "description": "Register an account and sign in to obtain a bearer token.",
    },
    {
        "name": "Documents",
        "description": "Upload supported documents for chunking, embedding, and Qdrant indexing.",
    },
    {
        "name": "Chat",
        "description": "Chat with the assistant and continue conversations using their IDs.",
    },
    {
        "name": "Usage",
        "description": "Administrator-only cumulative LLM token usage reports.",
    },
]

app = FastAPI(
    title="ETIC GenAI API",
    summary="Document ingestion and conversational AI endpoints.",
    description=(
        "Register and sign in, upload documents, or chat with the assistant. "
        "Interactive Swagger UI is available at `/docs`; the OpenAPI schema is at `/openapi.json`."
    ),
    version="0.1.0",
    openapi_tags=API_TAGS,
    contact={"name": "ETIC GenAI"},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "127.0.0.1:52766",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(usage_router)
