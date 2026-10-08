"""VTK MCP client for vtk-prompt integration.

Provides a thin wrapper around a vtk-mcp server, reached either over HTTP
(an already-running server: docker compose, manual ``vtk-mcp --transport
http``) or over stdio (a subprocess vtk-prompt spawned itself via
``--embed-mcp``, see ``mcp_launcher.py``). Provides:
- Vector search over VTK code examples and documentation
- VTK class API documentation lookup (query enrichment)
- Full VTK code validation via vtk-mcp
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

import requests

from . import get_logger

logger = get_logger(__name__)

DEFAULT_READ_TIMEOUT = 10.0


class VTKMCPClient:
    """Client for vtk-mcp server, over HTTP or stdio."""

    def __init__(
        self,
        base_url: str | None = "http://localhost:8000",
        process: subprocess.Popen | None = None,
        startup_timeout: float = DEFAULT_READ_TIMEOUT,
    ) -> None:
        """Initialize the client and perform the MCP handshake.

        Talks stdio to ``process`` (a live vtk-mcp subprocess) when given,
        otherwise HTTP to ``base_url``. ``ready`` tells whether the handshake
        succeeded.
        """
        self.base_url = base_url
        self._process = process
        self._lock = threading.Lock()  # one request/reply at a time on the pipe
        self._lines: queue.Queue[str] = queue.Queue()
        if process is not None:
            # A reader thread keeps timeouts simple; select() on a buffered
            # text stream can miss lines already sitting in Python's buffer.
            threading.Thread(target=self._read_stdout, daemon=True).start()
        self._session_id: str | None = None
        self._req_id = 0
        self.ready = False
        self._initialize_session(read_timeout=startup_timeout)

    def _next_id(self) -> str:
        self._req_id += 1
        return str(self._req_id)

    def _initialize_session(self, read_timeout: float) -> None:
        """Perform MCP JSON-RPC handshake."""
        data = self._request(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "vtk-prompt", "version": "1.0.0"},
                },
            },
            read_timeout=read_timeout,
        )
        if data is not None:
            self.ready = True
            self._request({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def _read_stdout(self) -> None:
        """Feed the subprocess's stdout lines to the queue; "" marks EOF."""
        assert self._process is not None
        for line in iter(self._process.stdout.readline, ""):  # type: ignore[union-attr]
            self._lines.put(line)
        self._lines.put("")

    def _request(self, payload: dict, read_timeout: float = DEFAULT_READ_TIMEOUT) -> dict | None:
        """Send a JSON-RPC request/notification; return the parsed response (or None)."""
        if self._process is not None:
            return self._stdio_request(payload, read_timeout)
        return self._http_request(payload)

    def _http_request(self, payload: dict) -> dict | None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        try:
            resp = requests.post(f"{self.base_url}/mcp/", json=payload, headers=headers, timeout=10)
        except Exception as e:
            logger.debug("MCP request failed: %s", e)
            return None
        if not self._session_id:
            self._session_id = resp.headers.get("Mcp-Session-Id")
        try:
            for line in resp.text.strip().split("\n"):
                if line.startswith("data: "):
                    return json.loads(line[6:])
        except Exception as e:
            logger.debug("MCP response parse error: %s", e)
        return None

    def _stdio_request(self, payload: dict, read_timeout: float) -> dict | None:
        """Write one JSON-RPC message to the subprocess and read its matching reply."""
        assert self._process is not None
        with self._lock:
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")  # type: ignore[union-attr]
                self._process.stdin.flush()  # type: ignore[union-attr]
            except Exception as e:
                logger.debug("MCP stdio write failed: %s", e)
                return None
            if "id" not in payload:
                return None  # notification: no reply expected
            # Skip lines that are not the reply to this request (server
            # notifications, or a late reply to an earlier request that timed out).
            deadline = time.monotonic() + read_timeout
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    break
                if not line:
                    return None  # subprocess closed stdout
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if isinstance(msg, dict) and str(msg.get("id")) == str(payload["id"]):
                    return msg
            logger.debug("MCP stdio read timed out after %ss", read_timeout)
            return None

    def _call_tool(self, name: str, arguments: dict) -> str | None:
        """Call a tool on the MCP server and return the text result."""
        data = self._request(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if not data:
            return None
        content = data.get("result", {}).get("content", [])
        if content:
            return content[0].get("text")
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
        data = self._request({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"})
        if not data:
            return []
        tools = data.get("result", {}).get("tools", [])
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

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

    def translate_prompt(
        self,
        query: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> str | None:
        """Translate a natural language query into the VTK pipeline DSL.

        Args:
            query: Natural language prompt.
            model: LiteLLM model override (e.g. ``ollama/llama3``).
            base_url: Base URL for OpenAI-compatible endpoints (e.g. Ollama).
            api_key: API key for the endpoint.

        Returns the DSL string, or None if the tool call fails.
        """
        args: dict = {"query": query}
        if model:
            args["model"] = model
        if base_url:
            args["base_url"] = base_url
        if api_key:
            args["api_key"] = api_key
        result = self._call_tool("translate_prompt_to_dsl", args)
        if not result or result.startswith("Error:"):
            logger.warning("DSL translation failed: %s", result)
            return None
        return result

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
        hints = [ctx for name in class_names[:3] if name and (ctx := self.get_class_context(name))]
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
