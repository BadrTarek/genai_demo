# etic_genai_demo

AI agents built with [LangChain](https://python.langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai]"      # or .[anthropic], or your provider's package
cp .env.example .env                # then set LLM_PROVIDER, LLM_MODEL and the API key
pytest                              # tests use a fake LLM, no API key needed
python scripts/run_agent.py         # chat with the example agent
uvicorn etic_genai.api.main:app --reload  # start the HTTP API
```

The API provides `POST /auth/register` and `POST /auth/sign-in`. Registration
requires a unique email and a password of at least 12 characters with uppercase,
lowercase, number, and special character. Sign-in returns a bearer token. Use
the **Authorize** button in Swagger to provide it, then call the protected
`POST /documents/upload` or `POST /chat` endpoints. Uploads are scoped to the
signed-in user, and chat retrieves only that user's documents. Reuse the
returned `conversation_id` to continue a saved conversation.
Open `http://127.0.0.1:8000/docs` to try the endpoints interactively.
Document uploads are private by default. Set the upload form field
`visibility=public` to let other users retrieve that document only as a fallback
when their own document search returns no chunks.

SQLite stores account data, expiring sessions, and LangGraph checkpoints in
`SQLITE_DB_PATH` (default `./data/documents.db`). Passwords are stored as
scrypt hashes, not plaintext. Five failed sign-in attempts within 15 minutes
lock the account for 30 minutes. Sessions expire after 30 minutes of inactivity.
Use HTTPS in deployments beyond local development.
The chat endpoint embeds each question and combines dense semantic search with
BM25 sparse search using Qdrant's reciprocal rank fusion (RRF), returning up to
five chunks from the same collection used by document upload. FastEmbed's
`Qdrant/bm25` encodes BM25-normalized passage term weights and query terms;
Qdrant applies corpus IDF through the sparse-vector index modifier. Its
response includes the retrieved source metadata. New collections are created
with dense and BM25 sparse vectors. Existing points created with the previous
hashed-token sparse encoding must be cleared and re-uploaded or re-indexed
because their token IDs are incompatible with BM25. An embedded collection with
an incompatible schema (dense-only or a different sparse-vector name) is
recreated automatically only when empty; if points remain, startup fails with
instructions to clear the collection before re-indexing. Existing remote
collections are updated with the sparse-vector configuration, but their points
still need re-uploading or re-indexing to gain BM25 vectors.

### LLM token usage

Each chat request records model-reported input and output tokens cumulatively
per `(user_id, conversation_id)` in the `conversation_usage` table in the same
SQLite database. Usage from every completed model call in an agent run is
counted. `unreported_calls` identifies model calls whose provider integration
did not return token usage, so totals may be incomplete for those calls.
Embedding usage is not included.

The administrator-only `GET /admin/usage` endpoint lists totals for all users
and conversations. Accounts are regular users by default. To promote a trusted
account, stop the API and run this SQL against the configured SQLite database:

```sql
UPDATE users SET is_admin = 1 WHERE email = 'admin@example.com';
```

Restart the API; the promotion applies to the account's next authorized
request. The `is_admin` column is added automatically to existing databases;
existing accounts remain non-admin.

The administrator-only `POST /admin/evaluations/run` endpoint runs the
`brd-retrieval-evaluation` LangSmith dataset using the signed-in administrator's
own documents. It returns the experiment name, ID, and LangSmith URL. Configure
`LANGSMITH_API_KEY` and ensure the dataset already exists before running an
evaluation. Evaluations are synchronous and may take several minutes.
Administrators can also run an evaluation from the **Evaluations** page in the
frontend workspace.

### Evaluate retrieval and answers

Run `python tests/evaluate_agent.py` to evaluate the end-to-end chat retrieval
and assistant response in LangSmith. Each experiment reports
`retrieval_relevance` for the retrieved context and `answer_correctness` for
the assistant response against the example's `expected_context` reference
facts. Evaluation examples need `question`, `user_id`, and non-empty
`expected_context` fields.

### View indexed documents in the Qdrant dashboard

The default setup uses embedded Qdrant storage and does not expose the Qdrant
dashboard. To run Qdrant as a local server with persistent Docker storage, start
Docker Desktop and run this PowerShell command:

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Set the following in `.env` to connect the API and ingestion script to that
server instead of the embedded database. `QDRANT_MODE` must be `docker` to
select the server; `local` selects embedded storage:

```dotenv
QDRANT_MODE=docker
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=business_requirements
```

Open `http://localhost:6333/dashboard` to view collections and points. The
embedded database under `data/qdrant` is separate from the server's Docker
volume; upload the documents again through `/docs` or run `python run_script.py`
to populate the server collection. Stop the Docker container with
`docker stop qdrant`; start it again with `docker start qdrant`.

To switch back to embedded local storage, set `QDRANT_MODE=local`. The setting
is shared by the CLI script and API. Restart the API after changing `.env`.

## Project layout

```
src/etic_genai/
├── config/        Settings loaded from env / .env (pydantic-settings)
├── core/          LLM factory (get_llm), logging, exceptions
├── agents/        Agents: BaseAgent + concrete agents (one file per agent)
├── graphs/        Graph state schemas (BaseAgentState) and shared subgraphs
├── nodes/         Reusable graph nodes: BaseNode, LLMNode
├── tools/         Tools: BaseAgentTool, ToolRegistry; common/ for shared tools
├── prompts/       Prompt classes (BasePrompt); prompt text in templates/*.md
├── memory/        Checkpointers (conversation persistence)
├── api/           FastAPI app, schemas, dependencies, routes, and chat service
├── usage/         Model-token callback collection and SQLite conversation totals
├── retrievers/    RAG: loaders, splitters, vector stores, retrievers
└── utils/         Small shared helpers
tests/             pytest; conftest.py provides a fake LLM
scripts/           Runnable entry points
notebooks/         Experiments (don't import notebooks from src)
```

## How to add things

**A tool:** subclass `BaseAgentTool` in `tools/` (or `tools/<domain>/`), define an
`args_schema` pydantic model, implement `_execute`. Register it in
`tools/__init__.py` if other agents should be able to look it up by name.
See `tools/common/datetime_tool.py`.

**A prompt:** add `prompts/templates/<name>.md` and a `BasePrompt` subclass with
`template_name = "<name>"`. Keep prompt text out of Python code.

**A node:** subclass `BaseNode`, set `name`, implement
`run(state, config) -> dict` that returns only the state keys it changes.

**An agent:** subclass `BaseAgent`, set `name`, and implement `build_graph()`
returning an uncompiled `StateGraph(self.state_schema)`. Override
`default_tools()` and `state_schema` as needed. See `agents/assistant_agent.py`.

```python
from etic_genai.agents import AssistantAgent
from etic_genai.memory import get_checkpointer

agent = AssistantAgent(checkpointer=get_checkpointer())
agent.invoke("What time is it in Cairo?", thread_id="demo")
```

Use `get_sqlite_checkpointer` in services that need conversations to survive
process restarts; the API uses it automatically with `SQLITE_DB_PATH`.

## Conventions

- Never construct chat models directly; use `core.llm.get_llm()` so the provider stays configurable.
- Every module logs through `core.logging.get_logger(__name__)`.
- Raise errors from `core.exceptions`.
- Add a test with the fake LLM (`fake_llm_factory` fixture) for every agent.
- Lint and format with `ruff check . && ruff format .`.
