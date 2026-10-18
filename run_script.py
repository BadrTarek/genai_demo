"""Index, search, chat with or evaluate the knowledge base, or serve it over MCP.

Examples:
    python run_script.py index --path requirements [--force] [--reset]
    python run_script.py search --query "What are the user roles?"
    python run_script.py docs
    python run_script.py ask --question "Who can approve an invoice worth 5,000 EUR?"
    python run_script.py chat
    python run_script.py eval [--dataset evals/rag_dataset.jsonl]
    python run_script.py mcp
"""

import argparse
import json
from pathlib import Path

from etic_genai.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Knowledge base, RAG agent and MCP server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Chunk, embed, and index a directory")
    index_parser.add_argument("--path", default=settings.knowledge_base_path)
    index_parser.add_argument("--output", default=settings.index_output_path)
    index_parser.add_argument("--force", action="store_true", help="Re-embed unchanged files")
    index_parser.add_argument("--reset", action="store_true", help="Drop the collection first")

    search_parser = subparsers.add_parser("search", help="Search indexed document chunks")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=settings.max_search_results)

    subparsers.add_parser("docs", help="List indexed documents")

    ask_parser = subparsers.add_parser("ask", help="Ask the guarded RAG agent one question")
    ask_parser.add_argument("--question", required=True)

    subparsers.add_parser("chat", help="Interactive chat with the RAG agent")

    eval_parser = subparsers.add_parser("eval", help="Evaluate the RAG agent on a dataset")
    eval_parser.add_argument("--dataset", default="evals/rag_dataset.jsonl")
    eval_parser.add_argument("--output", default="output/eval_report.json")

    subparsers.add_parser("mcp", help="Start the MCP server over stdio")
    return parser


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "mcp":
        from etic_genai.mcp_server import main as run_mcp_server

        run_mcp_server()
        return

    from etic_genai.documents.indexing.knowledge_base import KnowledgeBase

    knowledge_base = KnowledgeBase.from_settings()

    if args.command == "index":
        if args.reset:
            knowledge_base.reset()
        print_json(knowledge_base.index_path(args.path, args.output, force=args.force))
    elif args.command == "search":
        print_json(knowledge_base.search(args.query, args.top_k))
    elif args.command == "docs":
        print_json(knowledge_base.list_documents())
    elif args.command == "ask":
        from etic_genai.agents import RAGAgent

        print_json(RAGAgent.summarize(RAGAgent(knowledge_base).invoke(args.question)))
    elif args.command == "chat":
        chat(knowledge_base)
    elif args.command == "eval":
        evaluate(knowledge_base, args.dataset, args.output)


def chat(knowledge_base) -> None:
    from etic_genai.agents import RAGAgent
    from etic_genai.memory import get_checkpointer

    agent = RAGAgent(knowledge_base, checkpointer=get_checkpointer())
    print(f"Chatting with '{agent.name}'. Ctrl+C to exit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        summary = RAGAgent.summarize(agent.invoke(question, thread_id="cli"))
        print(f"\n{agent.name}> {summary['answer']}")
        scores = {name: e["score"] for name, e in summary["evaluations"].items()}
        print(f"\n  status={summary['status']} attempts={summary['attempts']} scores={scores}")
        for event in summary["guardrail_events"]:
            print(f"  guardrail[{event['stage']}] {event['name']}: {event['reason']}")


def evaluate(knowledge_base, dataset: str, output: str) -> None:
    from etic_genai.agents import RAGAgent
    from etic_genai.evaluation.runner import (
        EvaluationRunner,
        default_evaluators,
        format_report,
        load_dataset,
    )

    # Online evaluators are off: the offline runner scores everything itself.
    agent = RAGAgent(knowledge_base, online_evaluators=[])
    runner = EvaluationRunner(agent, default_evaluators(agent.judge_llm))
    report = runner.run(load_dataset(dataset))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    print(f"\nFull report: {output}")


if __name__ == "__main__":
    main()
