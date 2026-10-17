"""MCP server for project tools and agents.

This module adapts existing project abstractions instead of reimplementing
business logic inside MCP handlers.
"""

from typing import Any

from etic_genai.agents import AssistantAgent
from etic_genai.core.exceptions import ConfigurationError
from etic_genai.tools import tool_registry


def call_registered_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a LangChain tool from the shared registry and normalize its result."""
    tool = tool_registry.get(name)
    return str(tool.invoke(arguments))


def ask_assistant(message: str, thread_id: str = "mcp") -> str:
    """Run the default assistant agent and return its latest answer."""
    agent = AssistantAgent()
    result = agent.invoke(message, thread_id=thread_id)
    return str(result["messages"][-1].content)


def create_mcp_server(name: str = "etic-genai"):
    """Create the MCP server.

    The import stays inside this function so the main package can be installed
    and tested without the optional MCP dependencies.
    """
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise ConfigurationError(
            "MCP support is not installed. Install it with: "
            'python -m pip install -e ".[mcp]"'
        ) from exc

    mcp = MCPServer(name)

    @mcp.tool()
    def current_datetime(timezone: str = "UTC") -> str:
        """Get the current date and time in an IANA timezone, for example Africa/Cairo."""
        return call_registered_tool("current_datetime", {"timezone": timezone})

    @mcp.tool()
    def request_document_access(provider: str, purpose: str) -> str:
        """Prepare a user approval request for Google Drive or SharePoint access."""
        return call_registered_tool(
            "request_document_access",
            {"provider": provider, "purpose": purpose},
        )
    @mcp.tool()
    def sharepoint_auth_start(user_id: str) -> str:
        """Start Microsoft login for one app user and return the verification URL and code."""
        return call_registered_tool("sharepoint_auth_start", {"user_id": user_id})

    @mcp.tool()
    def sharepoint_auth_finish(user_id: str) -> str:
        """Finish Microsoft login for the given user."""
        return call_registered_tool("sharepoint_auth_finish", {"user_id": user_id})
    @mcp.tool()
    def sharepoint_tenant_info(user_id: str) -> str:
        """Get the Microsoft tenant ID for a given user."""
        return call_registered_tool("sharepoint_tenant_info", {"user_id": user_id})
    
    @mcp.tool()
    def sharepoint_list_files(
        approved: bool,
        user_id:str,
        site_id: str | None = None,
        drive_id: str | None = None,
        folder_path: str = "",
        limit: int = 20,
    ) -> str:
        """List SharePoint files after user approval using Microsoft Graph."""
        return call_registered_tool(
            "sharepoint_list_files",
            {
                "approved": approved,
                "user_id":user_id,
                "site_id": site_id,
                "drive_id": drive_id,
                "folder_path": folder_path,
                "limit": limit,
            },
        )
    @mcp.tool()
    def sharepoint_search(
        approved: bool,
        user_id: str,
        query: str,
        site_id: str | None = None,
        drive_id: str | None = None,
        limit: int = 20,
    ) -> str:
        """Search SharePoint files after user approval using Microsoft Graph."""
        return call_registered_tool(
            "sharepoint_search",
            {
                "approved": approved,
                "user_id": user_id,
                "query": query,
                "site_id": site_id,
                "drive_id": drive_id,
                "limit": limit,
            },
        )
    @mcp.tool()
    def chat_with_assistant(message: str, thread_id: str = "mcp") -> str:
        """Send a message to the default assistant agent."""
        return ask_assistant(message=message, thread_id=thread_id)

    return mcp
