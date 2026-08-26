"""Tests for the embedded vtk-mcp subprocess launcher."""

from unittest.mock import MagicMock, patch

import pytest

from vtk_prompt.mcp_launcher import _free_port, embedded_mcp_server


class TestFreePort:
    def test_returns_bindable_port(self):
        port = _free_port()
        assert 0 < port < 65536


class TestEmbeddedMcpServer:
    def test_raises_when_vtk_mcp_not_installed(self):
        with patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="not installed"):
                with embedded_mcp_server():
                    pass

    def test_yields_url_once_ready(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher.check_mcp_available", return_value=True),
            patch("vtk_prompt.mcp_launcher._free_port", return_value=12345),
        ):
            with embedded_mcp_server() as url:
                assert url == "http://127.0.0.1:12345"

        mock_proc.terminate.assert_called_once()

    def test_raises_if_process_exits_early(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher._free_port", return_value=12345),
        ):
            with pytest.raises(RuntimeError, match="exited early"):
                with embedded_mcp_server():
                    pass

    def test_raises_on_startup_timeout(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("vtk_prompt.mcp_launcher.importlib.util.find_spec", return_value=object()),
            patch("vtk_prompt.mcp_launcher.subprocess.Popen", return_value=mock_proc),
            patch("vtk_prompt.mcp_launcher.check_mcp_available", return_value=False),
            patch("vtk_prompt.mcp_launcher._free_port", return_value=12345),
            patch("vtk_prompt.mcp_launcher.time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="did not become ready"):
                with embedded_mcp_server(startup_timeout=0.01):
                    pass

        mock_proc.terminate.assert_called_once()
