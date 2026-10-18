# etic_genai_demo

A Retrieval-Augmented Generation (RAG) system over business-requirement documents, built with
[LangChain](https://python.langchain.com), [LangGraph](https://langchain-ai.github.io/langgraph/),
Qdrant and MCP. The same knowledge base is used in two ways:

1. **Through MCP** – GitHub Copilot (or any MCP client) indexes and searches the documents.
2. **Through a guarded agent** – a LangGraph agent answers with citations, protected by input and
   output guardrails and scored by evaluators online and offline.

## Architecture

```
                    ┌────────────── Knowledge base (documents/) ──────────────┐
 folder ──► loaders ──► chunker ──► embeddings ──► Qdrant  (incremental: content hash per doc,
 md/txt/csv/pdf/docx   (heading-aware)              ▲        stale chunks removed)
                    └───────────────────────────────┼─────────────────────────┘
                                                    │ search
              ┌─────────────────────────────────────┴──────────────────┐
              │                                                        │
   MCP server (mcp_server.py)                     RAG agent (agents/rag_agent.py)
   index_documents / search_documents             input_guardrails ─► retrieve ─► generate
   list_documents  / ask_knowledge_base ────────►       │                 │          │
              ▲                                         ▼                 ▼          ▼
     Copilot / any MCP client                       blocked          no context  output_guardrails
                                                                                  │  ▲ retry
                                                                                  ▼  │
                                                                     evaluate ─► finalize (answer + sources)
```

**Guardrails** (`guardrails/`)

| Stage  | Guardrail                  | Action                                                    |
|--------|----------------------------|-----------------------------------------------------------|
| input  | `InputLengthGuardrail`     | block empty / oversized questions                          |
| input  | `PromptInjectionGuardrail` | block jailbreak / instruction-override attempts            |
| input  | `PIIRedactionGuardrail`    | redact emails, phones, IBANs, cards before the LLM sees them |
| retrieve | relevance threshold      | off-topic questions get "not in the knowledge base"        |
| output | `PIIRedactionGuardrail`    | redact personal data in the answer                         |
| output | `CitationGuardrail`        | answer must cite existing passages, e.g. `[1]`             |
| output | `GroundednessGuardrail`    | LLM judge checks every claim is supported; regenerate once with feedback, then fall back to a safe answer |

**Evaluators** (`evaluation/`)

| Evaluator                    | Type          | Measures                                   | Used        |
|------------------------------|---------------|--------------------------------------------|-------------|
| `FaithfulnessEvaluator`      | LLM judge     | hallucination: is the answer supported?    | online (guardrail) + offline |
| `AnswerRelevanceEvaluator`   | LLM judge     | does the answer address the question?      | online + offline |
| `ContextRelevanceEvaluator`  | LLM judge     | did retrieval find the needed passages?    | offline     |
| `AnswerCorrectnessEvaluator` | LLM judge     | agreement with a reference answer          | offline     |
| `RetrievalRecallEvaluator`   | deterministic | expected source documents retrieved        | offline     |
| `RetrievalMRREvaluator`      | deterministic | rank of the first expected source          | offline     |
| `ExpectedStatusEvaluator`    | deterministic | answered / blocked / declined as expected  | offline     |

The labelled dataset is [evals/rag_dataset.jsonl](evals/rag_dataset.jsonl); it includes
adversarial cases (prompt injection, off-topic, PII) so guardrails are regression-tested too.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
cp .env.example .env                               # set OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
pytest                                             # offline: fake LLM, fake embeddings, in-memory Qdrant
```

## Usage

```bash
python run_script.py index --path requirements     # incremental; --force re-embeds, --reset empties first
python run_script.py search --query "What are the password requirements?"
python run_script.py docs                          # what is indexed
python run_script.py ask --question "Who can approve an invoice worth 5,000 EUR?"
python run_script.py chat                          # multi-turn chat with the guarded agent
python run_script.py eval                          # offline evaluation -> output/eval_report.json
python run_script.py mcp                           # MCP server over stdio
```

### Streamlit front end

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py               # http://localhost:8501
```

| Tab | What it shows |
|-----|---------------|
| **Chat** | Multi-turn chat. Each step streams live while the agent runs. Each answer has an *Inspect* panel: the agent graph with the executed path highlighted (a red node marks where a guardrail stopped it), per-step results and timings, the retrieved passages scored against the relevance threshold, guardrail events, and the online evaluation scores. |
| **Knowledge base** | Index a folder (incremental / force / reset), see indexed documents, and a search playground showing exactly what MCP `search_documents` returns. |
| **Evaluation** | Run the offline evaluation with a progress bar, KPI tiles, mean score and pass rate per metric, a per-case score table and a drill-down with the judges' reasoning. |
| **Architecture** | The agent graph and the guardrail / evaluator catalogue. |

The sidebar sliders (passages, relevance threshold, groundedness threshold, attempts, online
evaluation) rebuild the agent on the fly, so you can watch their effect immediately.

If tables fail with `pyarrow requires NumPy 2.0`, your environment pins NumPy 1.x
(e.g. for scikit-learn/matplotlib): `pip install "pyarrow<21"`.

### Copilot (VS Code)

[.vscode/mcp.json](.vscode/mcp.json) registers the server. Open Copilot Chat in **Agent** mode,
enable the `etic-knowledge-base` tools and ask e.g. *"Using the knowledge base, which role can
reopen an approved invoice?"*. Copilot calls `search_documents` and writes the answer itself, or
`ask_knowledge_base` to get the guarded agent's cited answer with its guardrail/eval report.
Restart the server from the MCP view after changing code or `.env`.

### Running MCP and the agent at the same time

The local Qdrant store (`QDRANT_PATH`) can only be opened by one process. When Copilot's MCP
server is running and you also want to use the CLI or the Streamlit app, run a Qdrant server:

```bash
docker compose up -d qdrant      # then QDRANT_URL=http://localhost:6333 in .env, and re-index
```

## Project layout

```
src/etic_genai/
├── config/        Settings loaded from env / .env (pydantic-settings)
├── core/          LLM factory (get_llm), logging (stderr), exceptions
├── documents/     loaders/ (md, txt, csv, pdf, docx), processing/ (chunker), indexing/ (KnowledgeBase, QdrantStore)
├── embeddings/    Embedding model factory
├── guardrails/    Input/output guardrails and the GuardrailPipeline
├── evaluation/    Evaluators (LLM judges + deterministic) and the dataset runner
├── agents/        BaseAgent, AssistantAgent (tool-calling example), RAGAgent
├── nodes/         Graph nodes: LLMNode, rag.py (RAG agent nodes)
├── graphs/        State schemas (BaseAgentState, RAGState)
├── prompts/       Prompt classes; text in templates/*.md (answer + judge prompts)
├── memory/        Checkpointers (conversation persistence)
├── tools/         Tools: BaseAgentTool, ToolRegistry
└── mcp_server.py  MCP server
app/               Streamlit front end (streamlit_app.py) and chart/graph helpers (viz.py)
evals/             Labelled evaluation dataset
tests/             pytest (offline)
```

## Conventions

- Never construct chat models directly; use `core.llm.get_llm()` so the provider stays configurable.
- Read configuration through `config.get_settings()`, not `os.getenv`.
- Every module logs through `core.logging.get_logger(__name__)`. Never `print` in library code:
  stdout is the MCP protocol channel.
- Prompt text (including judge prompts) lives in `prompts/templates/*.md`.
- Add a test with the fake LLM (`fake_llm_factory` / `fake_judge_factory`) for every agent.
- Lint and format with `ruff check . && ruff format .`.
