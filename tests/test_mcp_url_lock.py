"""Tests for the administrator-fixed vtk-mcp URL."""

from types import SimpleNamespace

import pytest
import yaml
from trame.app import get_server

from vtk_prompt.controllers import configuration
from vtk_prompt.vtk_prompt_ui import VTKPromptApp


def _state(**kw):
    return SimpleNamespace(
        **{
            "mcp_url": "",
            "top_k": 5,
            "log_tool_calls": False,
            "agentic_retrieval": False,
            "data_root": "",
            **kw,
        }
    )


class TestPersist:
    def test_user_settings_file_does_not_receive_a_locked_url(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yml"
        monkeypatch.setenv("VTK_PROMPT_CONFIG", str(cfg))
        cfg.write_text("top_k: 3\n")
        app = SimpleNamespace(state=_state(mcp_url="https://admin.example", mcp_locked=True))
        configuration.persist_global_settings(app)
        saved = yaml.safe_load(cfg.read_text())
        assert "mcp_url" not in saved and saved["top_k"] == 5

    def test_unlocked_url_is_saved(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yml"
        monkeypatch.setenv("VTK_PROMPT_CONFIG", str(cfg))
        cfg.write_text("top_k: 3\n")
        app = SimpleNamespace(state=_state(mcp_url="http://mine", mcp_locked=False))
        configuration.persist_global_settings(app)
        assert yaml.safe_load(cfg.read_text())["mcp_url"] == "http://mine"


@pytest.fixture(autouse=True)
def _no_prompt_client(monkeypatch):
    """Building the app creates an LLM client when a key is configured; not needed here."""
    monkeypatch.setattr("vtk_prompt.state.initializer.init_prompt_client", lambda app: None)


_servers = iter(range(1000))


def make_app(**kwargs):
    """Build the app on its own server; the default one registers its flags only once."""
    server = get_server(f"mcp-lock-test-{next(_servers)}", client_type="vue3")
    return VTKPromptApp(server=server, **kwargs)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "user.yml"
    path.write_text("model: anthropic/claude-sonnet-5\nmcp_url: http://from-user-config\n")
    return str(path)


class TestApp:
    def test_admin_url_overrides_config_file_and_is_locked(self, config_file):
        app = make_app(custom_prompt_file=config_file, mcp_url="https://admin.example")
        assert app.state.mcp_url == "https://admin.example"
        assert app.state.mcp_locked is True

    def test_env_var_is_used_when_no_flag(self, config_file, monkeypatch):
        monkeypatch.setenv("VTK_PROMPT_MCP_URL", "https://env.example")
        app = make_app(custom_prompt_file=config_file)
        assert app.state.mcp_url == "https://env.example" and app.state.mcp_locked is True

    def test_not_locked_without_admin_url(self, config_file, monkeypatch):
        monkeypatch.delenv("VTK_PROMPT_MCP_URL", raising=False)
        app = make_app(custom_prompt_file=config_file)
        assert app.state.mcp_url == "http://from-user-config"
        assert app.state.mcp_locked is False

    def test_client_cannot_change_a_locked_url(self):
        app = make_app(mcp_url="https://admin.example")
        app.state.mcp_url = "http://evil"
        app._on_mcp_url_change()
        assert app.state.mcp_url == "https://admin.example"
