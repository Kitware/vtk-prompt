"""VTK MCP HTTP client for vtk-prompt integration.

Provides a thin wrapper around the vtk-mcp HTTP server for:
- Vector search over VTK code examples and documentation
- VTK class API documentation lookup (query enrichment)
- Full VTK code validation via vtk-mcp
"""

from __future__ import annotations

import json

import requests

from . import get_logger

logger = get_logger(__name__)


class VTKMCPClient:
    """HTTP client for vtk-mcp server."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client and perform the MCP handshake."""
        self.base_url = base_url
        self._session_id: str | None = None
        self._req_id = 0
        self._initialize_session()

    def _next_id(self) -> str:
        self._req_id += 1
        return str(self._req_id)

    def _initialize_session(self) -> None:
        """Perform MCP JSON-RPC handshake."""
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "vtk-prompt", "version": "1.0.0"},
                },
            }
        )
        if resp:
            self._session_id = resp.headers.get("Mcp-Session-Id")
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _post(self, payload: dict) -> requests.Response | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            return requests.post(
                f"{self.base_url}/mcp/", json=payload, headers=headers, timeout=10
            )
        except Exception as e:
            logger.debug("MCP request failed: %s", e)
            return None

    def _call_tool(self, name: str, arguments: dict) -> str | None:
        """Call a tool on the MCP server and return the text result."""
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if not resp:
            return None
        try:
            for line in resp.text.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    content = data.get("result", {}).get("content", [])
                    if content:
                        return content[0].get("text")
        except Exception as e:
            logger.debug("MCP response parse error: %s", e)
        return None

    def vector_search(self, query: str, top_k: int = 5) -> str | None:
        """Search VTK examples using hybrid vector similarity search."""
        return self._call_tool("vector_search_examples", {"query": query, "k": top_k})

    def search_classes(self, query: str, limit: int = 5) -> list[str]:
        """Find VTK class names relevant to a query."""
        result = self._call_tool("vtk_search_classes", {"query": query, "limit": limit})
        if result:
            try:
                data = json.loads(result)
                if isinstance(data, list):
                    return [c if isinstance(c, str) else c.get("class_name", "") for c in data]
            except Exception:
                pass
        return []

    def get_class_context(self, class_name: str) -> str | None:
        """Build a concise prompt hint for a VTK class using synopsis and action phrase."""
        info_raw = self._call_tool("vtk_get_class_info", {"class_name": class_name})
        if not info_raw:
            return None
        try:
            info = json.loads(info_raw)
        except Exception:
            return None
        if info.get("found") is False or "error" in info:
            return None

        label = f"`{class_name}`"
        module = info.get("module")
        if module:
            label += f" (from `{module}`)"

        action_raw = self._call_tool("vtk_get_class_action_phrase", {"class_name": class_name})
        action_phrase = ""
        if action_raw:
            try:
                action_phrase = json.loads(action_raw).get("action_phrase", "")
            except Exception:
                pass

        synopsis = info.get("synopsis", "")

        suffix = " — ".join(filter(None, [action_phrase, synopsis]))
        return f"{label}: {suffix}" if suffix else label

    def list_tools(self) -> list[dict]:
        """Return all MCP tools as OpenAI-compatible function definitions."""
        resp = self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"})
        if not resp:
            return []
        try:
            for line in resp.text.strip().split("\n"):
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    tools = data.get("result", {}).get("tools", [])
                    return [
                        {
                            "type": "function",
                            "function": {
                                "name": t["name"],
                                "description": t.get("description", ""),
                                "parameters": t.get(
                                    "inputSchema", {"type": "object", "properties": {}}
                                ),
                            },
                        }
                        for t in tools
                    ]
        except Exception as e:
            logger.debug("Failed to list MCP tools: %s", e)
        return []

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call any MCP tool by name and return its text result."""
        return self._call_tool(name, arguments) or f"Tool '{name}' returned no result"

    def validate_code(self, code: str) -> str | None:
        """Run full VTK API validation; returns diagnostic summary string or None if clean."""
        result = self._call_tool("validate_vtk_code", {"source": code})
        if not result:
            return None
        try:
            data = json.loads(result)
            if data.get("status") == "ok":
                return None
            diagnostics = data.get("diagnostics", [])
            if not diagnostics:
                return None
            return "\n".join(f"- {d.get('message', str(d))}" for d in diagnostics)
        except Exception:
            return None

    def get_enriched_context(self, query: str, top_k: int = 5) -> str:
        """Build context for the LLM combining code examples, docs, and VTK class hints."""
        parts = []

        examples = self.vector_search(query, top_k=top_k)
        if examples:
            parts.append(examples)

        docs = self._call_tool("vector_search_docs", {"query": query, "k": 3})
        if docs:
            parts.append(docs)

        class_names = self.search_classes(query, limit=3)
        hints = [
            ctx
            for name in class_names[:3]
            if name and (ctx := self.get_class_context(name))
        ]
        if hints:
            parts.append("## Relevant VTK Classes\n\n" + "\n".join(f"- {h}" for h in hints))

        return "\n\n".join(parts)


def check_mcp_available(url: str = "http://localhost:8000") -> bool:
    """Check if a vtk-mcp server is reachable at the given URL."""
    try:
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False
