"""Chat with an agent from the terminal: `python scripts/run_agent.py`."""

from etic_genai.agents import AssistantAgent
from etic_genai.memory import get_checkpointer


def main() -> None:
    agent = AssistantAgent(checkpointer=get_checkpointer())
    thread_id = "cli"
    print(f"Chatting with '{agent.name}'. Ctrl+C to exit.")
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input:
            continue
        result = agent.invoke(user_input, thread_id=thread_id)
        print(f"{agent.name}> {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
