"""Tests for the VTKMCPClient transports."""

import io
import json
import time
from unittest.mock import MagicMock, patch

from vtk_prompt.vtk_mcp_client import VTKMCPClient


class _Pipe:
    """stdout that blocks until the test has nothing more to say, then EOFs."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def readline(self) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            time.sleep(0.5)
            return ""


def _stdio_client(replies: list[str]) -> VTKMCPClient:
    """Client wired to a fake subprocess whose stdout yields `replies`."""
    proc = MagicMock()
    proc.stdin = io.StringIO()
    proc.stdout = _Pipe([line + "\n" for line in replies])
    with patch.object(VTKMCPClient, "_initialize_session"):
        return VTKMCPClient(base_url=None, process=proc)


class TestStdio:
    def test_skips_lines_that_are_not_the_reply(self):
        client = _stdio_client(
            [
                "not json",
                json.dumps({"jsonrpc": "2.0", "method": "notifications/log"}),
                json.dumps({"jsonrpc": "2.0", "id": "99", "result": {"stale": True}}),
                json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}),
            ]
        )
        msg = client._request({"jsonrpc": "2.0", "id": "1", "method": "x"}, read_timeout=1)
        assert msg is not None and msg["result"] == {"ok": True}

    def test_timeout_returns_none(self):
        client = _stdio_client([])
        assert (
            client._request({"jsonrpc": "2.0", "id": "1", "method": "x"}, read_timeout=0.1) is None
        )

    def test_notification_expects_no_reply(self):
        client = _stdio_client([])
        assert client._request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


class TestHttp:
    def test_parses_sse_data_line_and_keeps_session_id(self):
        resp = MagicMock()
        resp.headers = {"Mcp-Session-Id": "abc"}
        resp.text = 'event: message\ndata: {"id": "1", "result": {"ok": 1}}\n'
        with (
            patch.object(VTKMCPClient, "_initialize_session"),
            patch("vtk_prompt.vtk_mcp_client.requests.post", return_value=resp),
        ):
            client = VTKMCPClient("http://localhost:8000")
            msg = client._request({"jsonrpc": "2.0", "id": "1", "method": "x"})
        assert msg == {"id": "1", "result": {"ok": 1}}
        assert client._session_id == "abc"
