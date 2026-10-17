"""Example agent: a tool-calling (ReAct-style) loop. Copy this as a starting point."""

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from etic_genai.agents.base import BaseAgent
from etic_genai.nodes import LLMNode
from etic_genai.prompts import AssistantPrompt
from etic_genai.tools import tool_registry


class AssistantAgent(BaseAgent):
    name = "assistant"
    description = "General-purpose assistant that can call tools."

    def default_tools(self) -> list[BaseTool]:
        return tool_registry.get_many(
            ["current_datetime", "request_document_access", "sharepoint_list_files","sharepoint_search"]
        )

    def build_graph(self) -> StateGraph:
        prompt = AssistantPrompt().to_chat_prompt(agent_name=self.name)
        llm_node = LLMNode(self.llm, prompt, self.tools)

        builder = StateGraph(self.state_schema)
        builder.add_node(llm_node.name, llm_node)
        builder.add_edge(START, llm_node.name)

        if self.tools:
            builder.add_node("tools", ToolNode(self.tools))
            builder.add_conditional_edges(llm_node.name, tools_condition)
            builder.add_edge("tools", llm_node.name)
        else:
            builder.add_edge(llm_node.name, END)
        return builder
