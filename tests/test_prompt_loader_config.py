"""Regression tests: a config file (no 'messages') must not be used as a prompt.

A file loaded via custom_prompt_file is only a *prompt* when it defines messages.
A settings-only config (model/base_url/mcp_url/top_k/...) must apply its settings
but leave custom_prompt_data as None so the client uses built-in prompt assembly.
Otherwise the client sends an empty messages list and the LLM returns HTTP 400.
"""

import types

import yaml

from vtk_prompt.utils.prompt_loader import load_custom_prompt_file


def _mk_app(path: str) -> types.SimpleNamespace:
    app = types.SimpleNamespace()
    app.custom_prompt_file = str(path)
    app.custom_prompt_data = None
    app.state = types.SimpleNamespace(
        error_message="",
        top_k=0,
        retry_attempts=0,
        mcp_url="",
        local_base_url="",
        use_cloud_models=True,
        tab_index=0,
        provider="",
        model="",
        local_model="",
        temperature_supported=True,
        temperature=0.0,
        max_tokens=0,
    )
    return app


def test_config_only_file_is_not_used_as_prompt(tmp_path):
    p = tmp_path / "config.yml"
    p.write_text(
        yaml.safe_dump(
            {
                "model": "local/qwen2.5-coder",
                "base_url": "http://x:11434/v1",
                "mcp_url": "http://localhost:8000",
                "top_k": 5,
                "retries": 3,
            }
        )
    )
    app = _mk_app(p)
    load_custom_prompt_file(app)

    # Not treated as a prompt ...
    assert app.custom_prompt_data is None
    # ... but the settings were still applied.
    assert app.state.use_cloud_models is False
    assert app.state.local_model == "qwen2.5-coder"
    assert app.state.local_base_url == "http://x:11434/v1"
    assert app.state.mcp_url == "http://localhost:8000"
    assert app.state.top_k == 5
    assert app.state.retry_attempts == 3


def test_prompt_file_with_messages_is_kept(tmp_path):
    p = tmp_path / "prompt.yml"
    p.write_text(
        yaml.safe_dump(
            {
                "model": "local/qwen2.5-coder",
                "messages": [{"role": "system", "content": "hi"}],
            }
        )
    )
    app = _mk_app(p)
    load_custom_prompt_file(app)

    assert app.custom_prompt_data is not None
    assert "messages" in app.custom_prompt_data
