"""Tests for custom config round-trip (model/base_url/mcp_url restore)."""

import types

from vtk_prompt.utils import prompt_loader


def _fake_app(data):
    app = types.SimpleNamespace()
    app.custom_prompt_file = None
    app.custom_prompt_data = data
    app.state = types.SimpleNamespace()
    return app


def test_loader_restores_local_ollama_and_mcp():
    cfg = {
        "model": "local/qwen2.5-coder-32b-ctx16",
        "base_url": "http://192.168.50.157:11434/v1",
        "mcp_url": "http://localhost:8000",
        "top_k": 5,
        "retries": 3,
        "modelParameters": {"temperature": 0.1, "max_tokens": 4000},
    }
    app = _fake_app(cfg)
    prompt_loader._process_model_configuration(app)
    prompt_loader._process_rag_and_generation_settings(app)
    prompt_loader._process_model_parameters(app)

    assert app.state.use_cloud_models is False
    assert app.state.tab_index == 1
    assert app.state.local_model == "qwen2.5-coder-32b-ctx16"
    assert app.state.local_base_url == "http://192.168.50.157:11434/v1"
    assert app.state.mcp_url == "http://localhost:8000"
    assert app.state.top_k == 5
    assert app.state.retry_attempts == 3


def test_loader_sets_mcp_url_for_cloud_model():
    cfg = {"model": "anthropic/claude-sonnet-4-6", "mcp_url": "http://localhost:8000"}
    app = _fake_app(cfg)
    prompt_loader._process_model_configuration(app)
    prompt_loader._process_rag_and_generation_settings(app)
    assert app.state.use_cloud_models is True
    assert app.state.provider == "anthropic"
    assert app.state.mcp_url == "http://localhost:8000"
