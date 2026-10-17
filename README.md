# etic_genai_demo

AI agents built with [LangChain](https://python.langchain.com) and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai]"      # or .[anthropic], or your provider's package
cp .env.example .env                # then set LLM_PROVIDER, LLM_MODEL and the API key
pytest                              # tests use a fake LLM, no API key needed
python scripts/run_agent.py         # chat with the example agent
```

### MCP server

Install the MCP extra when you want to expose project tools and agents to an MCP host:

```bash
pip install -e ".[mcp,openai]"
python scripts/run_mcp_server.py
```

The server currently exposes:

- `current_datetime`: wraps the shared `tools` registry entry.
- `chat_with_assistant`: sends a message to `AssistantAgent`.

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

## Conventions

- Never construct chat models directly; use `core.llm.get_llm()` so the provider stays configurable.
- Every module logs through `core.logging.get_logger(__name__)`.
- Raise errors from `core.exceptions`.
- Add a test with the fake LLM (`fake_llm_factory` fixture) for every agent.
- Lint and format with `ruff check . && ruff format .`.
