"""Tests for default config and .env auto-discovery."""

import os

import pytest

from vtk_prompt.utils import env_config


@pytest.fixture(autouse=True)
def _restore_environ():
    """load_dotenv_files mutates os.environ directly; snapshot and restore it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_explicit_env_var_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "my.yml"
    cfg.write_text("model: anthropic/x\n")
    monkeypatch.setenv(env_config.CONFIG_ENV_VAR, str(cfg))
    assert env_config.discover_config_file() == str(cfg)


def test_cwd_config_discovered(tmp_path, monkeypatch):
    monkeypatch.delenv(env_config.CONFIG_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "vtk-prompt.yml").write_text("model: anthropic/x\n")
    assert env_config.discover_config_file() == str(tmp_path / "vtk-prompt.yml")


def test_xdg_config_home_discovered(tmp_path, monkeypatch):
    monkeypatch.delenv(env_config.CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)  # no cwd config present
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    appdir = xdg / "vtk-prompt"
    appdir.mkdir(parents=True)
    (appdir / "config.yml").write_text("model: anthropic/x\n")
    assert env_config.discover_config_file() == str(appdir / "config.yml")


def test_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(env_config.CONFIG_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    assert env_config.discover_config_file() is None


def test_dotenv_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    (tmp_path / ".env").write_text('OPENAI_API_KEY="ollama"\n# c\nFOO=bar\n')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOO", raising=False)
    loaded = env_config.load_dotenv_files()
    assert str(tmp_path / ".env") in loaded
    assert os.environ["OPENAI_API_KEY"] == "ollama"
    assert os.environ["FOO"] == "bar"


def test_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    (tmp_path / ".env").write_text("OPENAI_API_KEY=fromfile\n")
    monkeypatch.setenv("OPENAI_API_KEY", "preexisting")
    env_config.load_dotenv_files()
    assert os.environ["OPENAI_API_KEY"] == "preexisting"
