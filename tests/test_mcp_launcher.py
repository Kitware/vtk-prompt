"""Tests for the embedded vtk-mcp subprocess launcher."""

import io
import time
from unittest.mock import MagicMock, patch

import pytest

from vtk_prompt.mcp_launcher import embedded_mcp_server
from vtk_prompt.vtk_mcp_client import VTKMCPClient


class _SilentPipe:
    """stdout that never produces a line (a hung server)."""

    def readline(self) -> str:
        time.sleep(0.5)
        return ""


def _mock_proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    return proc


class TestEmbeddedMcpServer:
    def test_raises_when_vtk_mcp_not_installed(self):
        with patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="not installed"):
                with embedded_mcp_server():
                    pass

    def test_yields_ready_client(self):
        mock_proc = _mock_proc()
        ready_client = VTKMCPClient.__new__(VTKMCPClient)
        ready_client.ready = True

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher.VTKMCPClient", return_value=ready_client),
        ):
            with embedded_mcp_server() as client:
                assert client is ready_client

        mock_proc.terminate.assert_called_once()

    def test_raises_if_process_exits_early(self):
        mock_proc = _mock_proc()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        not_ready_client = VTKMCPClient.__new__(VTKMCPClient)
        not_ready_client.ready = False

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher.VTKMCPClient", return_value=not_ready_client),
        ):
            with pytest.raises(RuntimeError, match="exited early"):
                with embedded_mcp_server():
                    pass

    def test_raises_on_startup_timeout(self):
        mock_proc = _mock_proc()
        not_ready_client = VTKMCPClient.__new__(VTKMCPClient)
        not_ready_client.ready = False

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher.VTKMCPClient", return_value=not_ready_client),
        ):
            with pytest.raises(RuntimeError, match="did not become ready"):
                with embedded_mcp_server(startup_timeout=0.01):
                    pass

        mock_proc.terminate.assert_called_once()


class TestVTKMCPClientStdio:
    def test_handshake_and_tool_call_over_stdio(self):
        proc = _mock_proc()
        proc.stdout = io.StringIO(
            '{"jsonrpc": "2.0", "id": "1", "result": {}}\n'
            '{"jsonrpc": "2.0", "id": "2", "result": {"content": [{"text": "hello"}]}}\n'
        )
        client = VTKMCPClient(base_url=None, process=proc)
        assert client.ready is True

        result = client._call_tool("vector_search_examples", {"query": "cube"})
        assert result == "hello"

    def test_not_ready_when_read_times_out(self):
        proc = _mock_proc()
        proc.stdout = _SilentPipe()
        client = VTKMCPClient(base_url=None, process=proc, startup_timeout=0.01)
        assert client.ready is False
